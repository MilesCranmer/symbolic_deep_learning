import jax
from jax import ops
from jax import numpy as np
from matplotlib import pyplot as plt
from jax import jit, vmap, grad, pmap
from jax.experimental.ode import odeint
from jax import random
import numpy as onp
import matplotlib as mpl
import multiprocessing
from functools import partial
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from celluloid import Camera
s_ = onp.s_
tqdm = lambda _: _

def make_transparent_color(ntimes, fraction):
  rgba = onp.ones((ntimes, 4))
  alpha = onp.linspace(0, 1, ntimes)[:, np.newaxis]
  color = np.array(mpl.colors.to_rgba(mpl.cm.gist_ncar(fraction)))[np.newaxis, :]
  rgba[:, :] = 1*(1-alpha) + color*alpha
  rgba[:, 3] = alpha[:, 0]
  return rgba

def get_potential(sim, sim_obj):

    dim = sim_obj._dim

    @jit
    def potential(x1, x2):
      """The potential between nodes x1 and x2"""
      dist = np.sqrt(np.sum(np.square(x1[:dim] - x2[:dim])))
      #Prevent singularities:
      min_dist = 1e-2
    #   bounded_dist = dist*(dist > min_dist) + min_dist*(dist <= min_dist)
      bounded_dist = dist + min_dist


    #   test_dist(jax.device_get(np.sum(np.any(dist <= min_dist))))

      if sim == 'r2':
          return -x1[-1]*x2[-1]/bounded_dist
      elif sim == 'r1':
          return x1[-1]*x2[-1]*np.log(bounded_dist)
      elif sim in ['spring', 'damped']:
          potential = (bounded_dist - 1)**2
          if sim == 'damped':
            damping = 1
            potential += damping*x1[1]*x1[1+sim_obj._dim]/sim_obj._n
            potential += damping*x1[0]*x1[0+sim_obj._dim]/sim_obj._n
            if sim_obj._dim == 3:
                potential += damping*x1[2]*x1[2+sim_obj._dim]/sim_obj._n

          return potential
      elif sim == 'string':
          return (bounded_dist - 1)**2 + x1[1]*x1[-1]
      elif sim == 'string_ball':
          potential = (bounded_dist - 1)**2 + x1[1]*x1[-1]
          r = np.sqrt((x1[1] + 15)**2 + (x1[0] - 5)**2)
          radius = 4
          potential += 10000/np.log(1+np.exp(10000*(r-radius)))#ball
          return potential

      elif sim in ['charge', 'superposition']:
          charge1 = x1[-2]
          charge2 = x2[-2]

          potential = charge1*charge2/bounded_dist
          if sim in ['superposition']:
              m1 = x1[-1]
              m2 = x2[-1]
              potential += -m1*m2/bounded_dist
        
          return potential
      elif sim in ['discontinuous']:
          m1 = x1[-1]
          m2 = x2[-1]
          q1 = x1[-2]
          q2 = x2[-2]
          pot_a = 0.0
          pot_b = 0.0 #-m1*m2/bounded_dist
          pot_c = (bounded_dist - 1)**2

          potential = (
            pot_a * (bounded_dist < 1) +
            (bounded_dist >= 1) * (
            pot_b * (bounded_dist < 2) +
            pot_c * (bounded_dist >= 2))
          )
          return potential
        

      else:
          raise NotImplementedError('No such simulation ' + str(sim))

    return potential

class SimulationDataset(object):

    """Docstring for SimulationDataset. """

    def __init__(self, sim='r2', n=5, dim=2,
            dt=0.01, nt=100, extra_potential=None,
            **kwargs):
        """TODO: to be defined.

        :sim: Simulation to run
        :n: number of bodies
        :nt: number of timesteps returned
        :dt: time step (can also set self.times later)
        :dim: dimension of simulation
        :pairwise: custom pairwise potential taking two nodes as arguments
        :extra_potential: function taking a single node, giving a potential
        :kwargs: other kwargs for sim

        """
        self._sim = sim
        self._n = n
        self._dim = dim
        self._kwargs = kwargs
        self.dt = dt
        self.nt = nt
        self.data = None
        self.times = np.linspace(0, self.dt*self.nt, num=self.nt)
        self.G = 1
        self.extra_potential = extra_potential
        self.pairwise = get_potential(sim=sim, sim_obj=self)

    def simulate(self, ns, key=0):
        rng = random.PRNGKey(key)
        vp = jit(vmap(self.pairwise, (None, 0), 0))
        n = self._n
        dim = self._dim 

        sim = self._sim
        # params = 1
        # if sim in ['charge']:
        #     params = 2
        params = 2
        total_dim = dim*2+params
        times = self.times
        G = self.G
        if self.extra_potential is not None:
          vex = vmap(self.extra_potential, 0, 0)

        @jit
        def total_potential(xt):
          sum_potential = np.zeros(())
          for i in range(n - 1):
            if sim in ['string', 'string_ball']:
                #Only with adjacent nodes
                sum_potential = sum_potential + G*vp(xt[i], xt[[i+1]]).sum()
            else:
                sum_potential = sum_potential + G*vp(xt[i], xt[i+1:]).sum()
          if self.extra_potential is not None:
            sum_potential = sum_potential + vex(xt).sum()
          return sum_potential

        @jit
        def force(xt):
          return -grad(total_potential)(xt)[:, :dim]

        @jit
        def acceleration(xt):
          return force(xt)/xt[:, -1, np.newaxis]

        unpacked_shape = (n, total_dim)
        packed_shape = n*total_dim


        @jit
        def odefunc(y, t):
          dim = self._dim
          y = y.reshape(unpacked_shape)
          a = acceleration(y)
          return np.concatenate(
              [y[:, dim:2*dim],
               a, 0.0*y[:, :params]], axis=1).reshape(packed_shape)

        @partial(jit, backend='cpu')
        def make_sim(key):
            if sim in ['string', 'string_ball']:
                x0 = random.normal(key, (n, total_dim))
                x0 = x0.at[..., -1].set(1) #const mass
                x0 = x0.at[..., 0].set(np.arange(n)+x0.at[...,0]*0.5)
                x0 = x0.at[..., 2:3].set(0.0)
            else:
                x0 = random.normal(key, (n, total_dim))
                x0 = x0.at[..., -1].set(np.exp(x0[..., -1])); #all masses set to positive
                if sim in ['charge', 'superposition']:
                    x0 = x0.at[..., -2].set(np.sign(x0[..., -2])); #charge is 1 or -1

            x_times = odeint(
                odefunc,
                x0.reshape(packed_shape),
                times, mxstep=2000).reshape(-1, *unpacked_shape)

            return x_times

        keys = random.split(rng, ns)
        vmake_sim = jit(vmap(make_sim, 0, 0), backend='cpu')
        # self.data = jax.device_get(vmake_sim(keys))
        # self.data = np.concatenate([jax.device_get(make_sim(key)) for key in keys])
        data = []
        for key in tqdm(keys):
            data.append(make_sim(key))
        self.data = np.array(data)

    def get_acceleration(self):
        vp = jit(vmap(self.pairwise, (None, 0), 0))
        n = self._n
        dim = self._dim 
        sim = self._sim
        params = 2
        total_dim = dim*2+params
        times = self.times
        G = self.G
        if self.extra_potential is not None:
          vex = vmap(self.extra_potential, 0, 0)
        @jit
        def total_potential(xt):
          sum_potential = np.zeros(())
          for i in range(n - 1):
            if sim in ['string', 'string_ball']:
                #Only with adjacent nodes
                sum_potential = sum_potential + G*vp(xt[i], xt[[i+1]]).sum()
            else:
                sum_potential = sum_potential + G*vp(xt[i], xt[i+1:]).sum()
          if self.extra_potential is not None:
            sum_potential = sum_potential + vex(xt).sum()
          return sum_potential

        @jit
        def force(xt):
          return -grad(total_potential)(xt)[:, :dim]

        @jit
        def acceleration(xt):
          return force(xt)/xt[:, -1, np.newaxis]

        vacc = vmap(acceleration, 0, 0)
        # ^ over time
        vacc2 = vmap(vacc, 0, 0)
        # ^ over batch
        return vacc2(self.data)

    def plot(self, i, animate=False, plot_size=True, s_size=1):
        #Plots i
        n = self._n
        times = onp.array(self.times)
        x_times = onp.array(self.data[i])
        sim = self._sim
        masses = x_times[:, :, -1]
        if not animate:
            if sim in ['string', 'string_ball']:
                rgba = make_transparent_color(len(times), 0)
                for i in range(0, len(times), len(times)//10):
                    ctimes = x_times[i]
                    plt.plot(ctimes[:, 0], ctimes[:, 1], color=rgba[i])
                plt.xlim(-5, 20)
                plt.ylim(-20, 5)
            else:
                for j in range(n):
                  rgba = make_transparent_color(len(times), j/n)
                  if plot_size:
                    plt.scatter(x_times[:, j, 0], x_times[:, j, 1], color=rgba, s=3*masses[:, j]*s_size)
                  else:
                    plt.scatter(x_times[:, j, 0], x_times[:, j, 1], color=rgba, s=s_size)
        else:
            if sim in ['string', 'string_ball']: raise NotImplementedError
            fig = plt.figure()
            camera = Camera(fig)
            d_idx = 20
            for t_idx in range(d_idx, len(times), d_idx):
                start = max([0, t_idx-300])
                ctimes = times[start:t_idx]
                cx_times = x_times[start:t_idx]
                for j in range(n):
                  rgba = make_transparent_color(len(ctimes), j/n)
                  if plot_size:
                    plt.scatter(cx_times[:, j, 0], cx_times[:, j, 1], color=rgba, s=3*masses[:, j])
                  else:
                    plt.scatter(cx_times[:, j, 0], cx_times[:, j, 1], color=rgba, s=s_size)
#                 plt.xlim(-10, 10)
#                 plt.ylim(-10, 10)
                camera.snap()
            from IPython.display import HTML
            return HTML(camera.animate().to_jshtml())


