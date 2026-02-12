# -*- coding: utf-8 -*-
# Introdução ao Aprendizado por Reforço - PPGEE
# Prof. Armando Alves Neto
########################################################################
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import numpy as np
import cv2
from functools import partial
import torch
from torch.autograd import Variable

import class_flow_field
import class_map

MAX_STEPS = 200

########################################
# Normaliza ações no ambiente
########################################
class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """
    def action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.0
        act_b = (self.action_space.high + self.action_space.low)/ 2.0
        return act_k * action + act_b

    def reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.0
        return act_k_inv * (action - act_b)

########################################
# classe do mapa
########################################
class Env(gym.Env):
	########################################
	# construtor
	def __init__(self, parameters, discrete_state=True, discrete_action=True, noise=0.0):
		
		################################################
		# define as posições dos nós sensores
		LVlist = []
		
		# going with flows
		#LVlist.append(np.array([1.5, 1.5]))
	
		#rz, delta, Gamma, omega = LV	
		LVlist.append([np.array([30,30]), 10, -50])
		LVlist.append([np.array([70,70]), 10,  50])
		LVlist.append([np.array([30,70]), 10,  50])
		LVlist.append([np.array([70,30]) + noise*np.ones(2), 10,  50])
		
		#LVlist = [np.array([-10, 585]), 17.5, 85.0], [np.array([105, 597]), 17.5, -85.0], [np.array([250, 650]), 17.5, -40.0], [np.array([400, 670]), 25.0, -100.0], [np.array([650, 650]), 26.25, -100.0], [np.array([187, 495]), 25.0, 90.0], [np.array([350, 550]), 25.0, 100.0], [np.array([650, 530]), 25.0, 95.0], [np.array([507, 498]), 23.75, -75.0], [np.array([405, 460]), 21.25, -70.0], [np.array([220, 430]), 17.5, -90.0], [np.array([ 50, 380]), 22.5, -95.0], [np.array([157, 303]), 25.0, 100.0], [np.array([510, 380]), 25.0, 100.0], [np.array([675, 410]), 18.75, -90.0], [np.array([670, 260]), 25.0, 80.0], [np.array([505, 285]), 17.5, -80.0], [np.array([380, 245]), 25.0, 105.0], [np.array([ 13, 150]), 25.0, -100.0], [np.array([150, 148]), 25.0, 95.0], [np.array([300, 150]), 16.25, -60.0], [np.array([550, 125]), 22.5, -80.0], [np.array([458,  65]), 22.5, 95.0], [np.array([730,  30]), 16.25, -70.0], [np.array([280, -15]), 25.0, 100.0], [np.array([-10, -50]), 25.0, 70.0], [np.array([251, 204]), 12.5, -35.0], [np.array([309, 332]), 10.0, -24.0]
		
		# field
		#print('Creating exposure map...')
		self.field = class_flow_field.Field(parameters, LVlist, dim = 2)
		#print('\nMap ready!')
		
		################################################
		# estados discretos?
		self.discrete_state = discrete_state
		# ações discretos?
		self.discrete_action = discrete_action
		
		# seta limites do ambiente
		self.xlim, self.ylim = [parameters.get(k) for k in ['xlim', 'ylim']]

		# resolucao
		self.vstill_norm = class_flow_field.VMAX	# Vstill(x, t) <= Vmax < Vfm
		self.nx = int(np.abs(np.diff(self.xlim))/self.vstill_norm)
		self.ny = int(np.abs(np.diff(self.ylim))/self.vstill_norm)
		self.num_states = [self.nx, self.ny]
		print(f'self.num_states = {self.num_states}')
			
		# espaço de observação
		if self.discrete_state:		
			# converte estados continuos em discretos
			lower_bounds = [self.xlim[0], self.ylim[0]]
			upper_bounds = [self.xlim[1], self.ylim[1]]
			self.get_state = partial(self.obs_to_state, self.num_states, lower_bounds, upper_bounds)
		else:
			self.state_list_low  = np.array([self.xlim[0], self.ylim[0]], dtype=np.float32)
			self.state_list_high = np.array([self.xlim[1], self.ylim[1]], dtype=np.float32)
			self.observation_space = spaces.Box(low = self.state_list_low, high = self.state_list_high, dtype=np.float32)
		
		# espaco de atuacao
		if self.discrete_action:
			self.nactions = 16
			self.action_space = spaces.Discrete(self.nactions)
			self.th = np.linspace(0.0, 2.0*np.pi, self.nactions+1)[:-1]
		else:
			# action box
			self.action_space = spaces.Box(low=0.0,  high=2.0*np.pi,  shape=(1,), dtype=np.float32)

		# cria mapa
		self.mapa = class_map.Map(parameters, img=parameters['map'])

		# alvo
		self.alvo = parameters['xgoal']
		
		# numero de episodios
		self.episodes = 0.0

	########################################
	# seed
	########################################
	def seed(self, rnd_seed = None):
		np.random.seed(rnd_seed)
		return [rnd_seed]

	########################################
	# reset
	########################################
	def reset(self):
		
		# numero de episodios
		self.episodes += 1
		
		# numero de passos
		self.steps = 0

		# posicao aleatória
		self.p = self.getRand()
		
		# trajetoria
		data = {'t':self.steps, 's':self.p, 'a':0}
		self.traj = [data]

		# estado inicial
		if self.discrete_state:
			return self.get_state(self.p), {}
		else:
			return self.p, {}

	########################################
	# converte acão para direção
	def actionU(self, action):
		
		if self.discrete_action:
			th = self.th[action]
		else:
			th = np.squeeze(action)
			
		return self.vstill_norm*np.array([np.cos(th), np.sin(th)])	
		
	########################################
	# step -> new_observation, reward, done, info = env.step(action)
	def step(self, action):
		
		# novo passo
		self.steps += 1
		
		# seleciona acao
		vnet = self.actionU(action)
		
		##############
		# estado antigo
		s = self.p

		# proximo estado
		sl = self.p + vnet

		# fora dos limites (norte, sul, leste, oeste) ou colisao
		if not self.collisionBetween(s, sl):
			self.p = sl
		
		# atualiza trajetoria
		data = {'t':self.steps, 's':self.p, 'a':action}
		self.traj.append(data)
		
		# reward
		reward = self.getReward(s, sl, vnet)
		
		# estado terminal?
		done = self.terminal()

		# retorna
		if self.discrete_state:
			return self.get_state(self.p), reward, done, {}, {}
		else:
			return self.p, reward, done, {}, {}

	########################################
	# função de reforço proposta
	def getReward(self, s, sl, vnet):
		
		# exposicao
		reward = 0.0
		m = int(np.ceil(class_flow_field.VMAX/self.field.Vfm)) + 2
		x = np.linspace(s[0], sl[0], m)
		y = np.linspace(s[1], sl[1], m)
		for i in range(m):
			p = np.array([x[i], y[i]])
			
			# se colisao, exposicao maxima
			if self.collision(p):
				reward -= self.field.kappa*(2.0*np.abs(self.field.Vfm)**self.field.alpha)
			else:
				reward -= self.exposure(p, vnet)
		
		if self.terminal():
			reward -= 5.0*np.linalg.norm(self.p - self.alvo)
		
		return reward
		
	########################################
	# função de reforço da literatura
	def getReward2(self, s, sl, vnet):
		
		reward = 0.0
		
		Rter = 100.0
		Robs = -100.0
		lc = 1.0
		Rdis = -5.0*np.linalg.norm(self.p - self.alvo)
		g = 0.1
		
		if self.goalReach():
			reward += Rter
			
		if self.collisionBetween(s,sl):
			reward += Robs
			
		if self.terminal():
			reward += Rdis
		
		reward += lc*(Rdis) + g*self.steps
		
		return reward
	
	########################################
	# chegou no alvo?
	def goalReach(self):
		if self.discrete_state:
			return (self.get_state(self.p) == self.get_state(self.alvo))
		else:
			return (np.linalg.norm(self.p - self.alvo) <= 2.0*self.vstill_norm)
			
	########################################
	# terminou?
	def terminal(self):
		
		# chegou no alvo
		if self.goalReach():
			return True
		
		# timeout
		if self.steps >= MAX_STEPS:
			return True
			
		return False
			
	########################################
	# define a função de exposição a partir de um conjunto de nós sensores e seus parâmetros
	########################################
	def exposure(self, p, u, dt=1.0, t=0.0):
		return self.field.Intensity(p, u, dt, t)

	########################################
	# pega ponto aleatorio no voronoi
	def getRand(self):
		
		# pega um ponto aleatorio
		while True:
			qx = np.random.uniform(self.xlim[0], self.xlim[1])
			qy = np.random.uniform(self.ylim[0], self.ylim[1])
			q = (qx, qy)
			
			# evita comecar muito proximo do alvo
			if np.linalg.norm(self.alvo - np.array(q)) < 1.0*self.vstill_norm:
				continue
			
			# verifica colisao
			if not self.collision(q):
				break

		# retorna
		return q

	########################################
	# colisao entre dois pontos
	def collisionBetween(self, qi, qf):
		m = int(np.ceil(class_flow_field.VMAX/self.field.Vfm)) + 2
		x = np.linspace(qi[0], qf[0], m)
		y = np.linspace(qi[1], qf[1], m)
		
		for i in range(m):
			if self.collision(np.array([x[i], y[i]])):
				return True
				
		return False
		
	########################################
	# verifica colisao com os obstaculos
	def collision(self, q):
		
		if not ( (self.xlim[0] <= q[0] <= self.xlim[1]) and (self.ylim[0] <= q[1] <= self.ylim[1]) ):
			return True
		
		if self.mapa.collision(q):
			return True
		
		return False

	##########################################
	# converte estados continuos em discretos
	def obs_to_state(self, num_states, lower_bounds, upper_bounds, obs):
		state_idx = []
		for ob, lower, upper, num in zip(obs, lower_bounds, upper_bounds, num_states):
			state_idx.append(self.discretize_val(ob, lower, upper, num))

		return np.ravel_multi_index(state_idx, num_states)

	##########################################
	# discretiza um valor
	def discretize_val(self, val, min_val, max_val, num_states):
		state = int(num_states * (val - min_val) / (max_val - min_val))
		if state >= num_states:
			state = num_states - 1
		if state < 0:
			state = 0
		return state

	########################################
	# desenha a imagem distorcida em metros
	def render(self, Q=[], arrows=True, device=None, path=True):
		
		FONTSIZE = 16
		
		# desenha o robo
		#plt.plot(self.p[0], self.p[1], 'rs')

		# desenha o alvo
		plt.plot(self.alvo[0], self.alvo[1], 'r', marker='x', markersize=20, linewidth=10)
		
		# trajetoria
		if path:
			px = [traj['s'][0] for traj in self.traj]
			py = [traj['s'][1] for traj in self.traj]
			plt.plot(px, py, 'm', linewidth=3)

		# plota mapa real e o mapa observado
		self.mapa.draw()

		# desenha o field
		self.field.draw(t=0, field=True)
		
		# vector field
		if arrows:
			xm = np.linspace(self.xlim[0], self.xlim[1], self.num_states[0])
			ym = np.linspace(self.ylim[0], self.ylim[1], self.num_states[1])
			XX, YY = np.meshgrid(xm, ym, indexing='ij')
			#
			vx = []
			vy = []
			for x in xm:
				for y in ym:
					# plota a melhor ação
					if self.discrete_state and self.discrete_action:
						s = self.get_state(np.array([x, y]))
						a = Q[s, :].argmax()
					elif self.discrete_action:
						s = torch.from_numpy(np.array([x,y])).float().unsqueeze(0).to(device)
						Q.eval()
						with torch.no_grad():
							action_values = Q(s)
						Q.train()
						a = np.argmax(action_values.cpu().data.numpy())
					else:
						s = torch.from_numpy(np.array([x,y])).float().unsqueeze(0).to(device)
						s = Variable(s)
						a = Q.forward(s)
						a = a.cpu().detach().numpy()[0,0]
					
					u = self.actionU(a)
					vx.append(u[0])
					vy.append(u[1])
			
			# plota setas
			plt.gca().quiver(XX, YY, np.array(vx), np.array(vy), angles='xy', scale_units='xy', scale=1.0, headwidth=5, alpha = 0.8)

		plt.xticks(fontsize=FONTSIZE)
		plt.yticks(fontsize=FONTSIZE)
		plt.xlim(self.xlim)
		plt.ylim(self.ylim)
		plt.box(True)
		plt.pause(.1)

	########################################
	def close(self):
		None
