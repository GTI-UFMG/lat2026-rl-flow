#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('./src')
import class_env as ce
import os
os.makedirs("./fields", exist_ok=True)
os.makedirs("./models", exist_ok=True)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import argparse

MEAN_REWARDS = 200
plt.rcParams['figure.figsize'] = (16,8)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

########################################################################
# Define arquitetura da Rede Neural Artificial (Perceptron) com ```state_size```entradas, ```action_size``` saídas e duas camadas escondidas de tamanho ```hidden```. Essa rede tem duas camadas escondidas com 256 neurônios cada, e basicamente recebe um estado $s$ na entrada, provendo na saída $\hat{q}(s,\cdot, \boldsymbol{w})$ para todas as ```action_size``` ações.
class QNetwork(nn.Module):
	##########################################
	def __init__(self, state_size, action_size, hidden=64):
		super(QNetwork, self).__init__()
		
		state_size  = int(state_size)
		action_size = int(action_size)
		hidden    = int(hidden)

		self.fc1 = nn.Linear(state_size, hidden)
		self.fc2 = nn.Linear(hidden, hidden)
		self.fc3 = nn.Linear(hidden, action_size)

	##########################################
	def forward(self, state):
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		return self.fc3(x)

########################################################################
# Define the Dueling DQN network
class DuelingNetwork(nn.Module):
	def __init__(self, state_size, action_size, hidden=64):
		super(DuelingNetwork, self).__init__()
		
		state_size  = int(state_size)
		action_size = int(action_size)
		hidden    = int(hidden)
		
		self.fc1 = nn.Linear(state_size, hidden)
		self.fc2 = nn.Linear(hidden, hidden)

		self.advantage_fc = nn.Linear(hidden, action_size)
		self.value_fc = nn.Linear(hidden, 1)

	##########################################
	def forward(self, x):
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))

		advantage = self.advantage_fc(x)
		value = self.value_fc(x)
		q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
		return q_values

########################################################################
# Gerenciando o replay buffer. O buffer tem tamanho fixo e armazena tuplas de experiências com o ambiente.
class ReplayBuffer:
	##########################################
	def __init__(self, parameters, buffer_size=int(1e5), seed=0):
		self.batch_size = parameters['batch_size']		
		self.D = deque(maxlen=buffer_size)  
		self.experience = namedtuple("Experience", field_names=["S", "A", "R", "Sl", "done"])
		self.seed = random.seed(seed)
		
	##########################################
	# Adiciona uma nova experiencia ao buffer
	def add(self, S, A, R, Sl, done):
		e = self.experience(S, A, R, Sl, done)
		self.D.append(e)
	
	##########################################
	# Escolhe aleatoriamente um mini-lote das experiencias
	def sample(self):
		batch_size = np.min([len(self.D), self.batch_size])
		Dj = random.sample(self.D, k=batch_size)
	
		states      = torch.from_numpy(np.vstack([e.S  for e in Dj if e is not None])).float().to(device)
		actions     = torch.from_numpy(np.vstack([e.A  for e in Dj if e is not None])).long().to(device)
		rewards     = torch.from_numpy(np.vstack([e.R  for e in Dj if e is not None])).float().to(device)
		next_states = torch.from_numpy(np.vstack([e.Sl for e in Dj if e is not None])).float().to(device)
		dones       = torch.from_numpy(np.vstack([e.done for e in Dj if e is not None]).astype(np.uint8)).float().to(device)
  
		return (states, actions, rewards, next_states, dones)
	
	##########################################
	def __len__(self):
		return len(self.D)

########################################################################
# Cria classe do algoritmo de DQN.
class DQN(object):
	_counter = 0
	
	def __init__(self, parameters):
		
		# conta as instancias
		self.id = DQN._counter
		DQN._counter += 1
		
		self.parameters = parameters

		# numero de episodios
		self.episode = 0
		
		# parametros de aprendizado
		self.gamma = parameters['gamma']
		self.eps = parameters['eps']
		self.alpha = parameters['alpha']
		self.update_rate = parameters['update_rate']
		self.tau = parameters['tau']
		self.type = parameters['type']
		
		# cria ambiente
		self.env = ce.Env(parameters, discrete_state=False)
		
		# log file (name depends on the method)
		self.logfile = 'models/' + parameters['q-file'] + "_" + self.type + f'[{self.id}]'
		
		# reseta
		self.reset()
	
	##########################################
	# reseta a rede
	def reset(self):
		
		# reseta o ambiente
		S, _ = self.env.reset()
		
		# Q-Networks		
		###################
		# DQN
		if self.type == 'DQN':
			self.Q       = QNetwork(state_size=self.env.observation_space.shape[0], action_size=self.env.action_space.n).to(device)
			self.Qtarget = QNetwork(state_size=self.env.observation_space.shape[0], action_size=self.env.action_space.n).to(device)
		###################	
		# Dueling DQN
		elif self.type == 'DuDQN':
			self.Q       = DuelingNetwork(state_size=self.env.observation_space.shape[0], action_size=self.env.action_space.n).to(device)
			self.Qtarget = DuelingNetwork(state_size=self.env.observation_space.shape[0], action_size=self.env.action_space.n).to(device)
		else:
			print('Erro da rede!')
			
		# otimizador
		self.optimizer = optim.Adam(self.Q.parameters(), lr=self.alpha)
		
		# Buffer de armazenamento de experiencias
		self.D = ReplayBuffer(self.parameters)
		
		# rewards
		self.rewards = []
		self.avg_rewards = []
		
		# carrega tabela pre-computada se for o caso
		if self.parameters['load_Q']:
			self.load()
		
	##########################################
	# politica epsilon-greedy 
	def act(self, S):

		S = torch.from_numpy(np.array(S)).float().unsqueeze(0).to(device)
		self.Q.eval()
		with torch.no_grad():
			action_values = self.Q(S)
		self.Q.train()

		# seleção de ação epsilon-greedy
		if np.random.random() > self.eps:
			return np.argmax(action_values.cpu().data.numpy())
		else:
			return np.random.choice(np.arange(self.env.action_space.n))
			
	########################################
	# salva rede Q
	def save(self):
		# salva rede
		model_scripted = torch.jit.script(self.Q) # Export to TorchScript
		model_scripted.save(self.logfile + '.pkl') # Save
		
		# outro parametros
		with open(self.logfile + '.npy', 'wb') as f:			
			np.savez(f, episodes=self.episode, rewards=self.rewards)

	########################################
	# carrega rede Q
	def load(self):
		
		# carrega rede
		self.Q = torch.jit.load(self.logfile + '.pkl')
		self.Q.eval()
		
		# outros dados
		with open(self.logfile+'.npy', 'rb') as f:
			data = np.load(f)
			self.episode = data['episodes']
			self.rewards = list(data['rewards'])
			self.avg_rewards = []
			for i in range(1,len(self.rewards)):
				a = self.rewards[:i]
				self.avg_rewards.append(np.mean(a[-MEAN_REWARDS:]))

	########################################################
	#Soft update model parameters
	def soft_update(self):
		for target_param, local_param in zip(self.Qtarget.parameters(), self.Q.parameters()):
			target_param.data.copy_( self.tau*local_param.data + (1.0-self.tau)*target_param.data )
			
	##########################################
	# Executa um episódio do DQN
	def runEpisode(self):
	
		# novo episodio
		self.episode += 1
		
		rewards = []
		
		# determina o estado inicial
		S, _ = self.env.reset()

		while True:
			
			# Escolhe uma ação (epsilon-greedy)
			A = self.act(S)
			
			# step
			Sl, R, done, _, _ = self.env.step(A)
			rewards.append(R)
			
			# Armazena a experiencia (S,A,R,Sl) no buffer D
			self.D.add(S, A, R, Sl, done)
			
			# Seleciona um mini-lote Dj de D e aprende
			Ss, As, Rs, Sls, dones = self.D.sample()
		
			# valor esperado da rede
			qhat = self.Q(Ss).gather(1, As)
			
			# Calcula o alvo da função valor-ação
			qtargets = self.Qtarget(Sls).detach().max(1)[0].unsqueeze(1)

			# Função de perda (erro médio quadrático)
			loss = F.mse_loss(qhat, Rs + (self.gamma*qtargets*(1-dones)))
			
			# aprendendo
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
				
			# aprende eventualmente
			if (self.episode % self.update_rate == 0):
				# atualiza rede alvo
				self.soft_update()

			# proximo estado
			S = Sl
			
			if done: 
				break
		
		# renderiza eventualmente
		if (self.episode % MEAN_REWARDS) == 0:
			# render
			if self.id == 0:
				plt.subplot(1, 2, 1)
				plt.gca().clear()
				self.env.render(self.Qtarget, device=device)
			
			# salva a tabela Q
			if self.parameters['save_Q']:
				self.save()
				
		# rewards
		self.rewards.append(np.sum(rewards))
		self.avg_rewards.append(np.mean(self.rewards[-MEAN_REWARDS:]))

########################################################################
def parse_args():
	parser = argparse.ArgumentParser(description="DQN Training Parameters")

	parser.add_argument('--episodes', type=int, default=1000)
	parser.add_argument('--gamma', type=float, default=0.99)
	parser.add_argument('--eps', type=float, default=0.1)
	parser.add_argument('--alpha', type=float, default=1.0e-3)
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--update_rate', type=int, default=5)
	parser.add_argument('--tau', type=float, default=1.0e-3)

	parser.add_argument('--type', type=str, default='DQN',
						choices=['DQN', 'DuDQN'])

	parser.add_argument('--save_Q', action='store_true')
	parser.add_argument('--load_Q', action='store_true')

	parser.add_argument('--resolution', type=int, default=500)
	parser.add_argument('--q_file', type=str, default='qnet')
	parser.add_argument('--map', type=str, default='imgs/islands.png')

	parser.add_argument('--xgoal', type=float, nargs=2, default=[110.0, 90.0])
	parser.add_argument('--xlim', type=float, nargs=2, default=[0.0, 120.0])
	parser.add_argument('--ylim', type=float, nargs=2, default=[0.0, 100.0])

	parser.add_argument('--nagents', type=int, default=2)

	return parser.parse_args()

########################################################################
# Programa principal:
# 
# - episodes: número de episódios
# - max_iter: máximo de iterações até o fim do episódio
# - gamma: fator de desconto
# - eps: $\varepsilon$
# - alpha: $\alpha$
# - batch_size: tamanho do mini-lote do buffer de experiencias
# - device: usa *cpu* ou *gpu*

if __name__ == '__main__':
	
	plt.ion()
	
	args = parse_args()

	parameters = {
		'episodes'   : args.episodes,
		'gamma'      : args.gamma,
		'eps'        : args.eps,
		'alpha'      : args.alpha,
		'batch_size' : args.batch_size,
		'update_rate': args.update_rate,
		'tau'        : args.tau,
		'type'       : args.type,
		'save_Q'     : args.save_Q,
		'load_Q'     : args.load_Q,
		'resolution' : args.resolution,
		'q-file'     : args.q_file,
		'map'        : args.map,
		'xgoal'      : np.array(args.xgoal),
		'xlim'       : np.array(args.xlim),
		'ylim'       : np.array(args.ylim),
		'nagents'    : args.nagents,
	}
	
	dqn = [DQN(parameters) for _ in range(parameters['nagents'])]
	
	avg_reward = []
	std_reward = []
	
	while dqn[0].episode <= parameters['episodes']:
		
		# roda um episodio
		for d in dqn:
			d.runEpisode()
			
		avg_reward.append(np.mean([d.rewards[-1] for d in dqn]))
		std_reward.append(np.std([d.rewards[-1] for d in dqn]))
		
		# print e plot
		if dqn[0].episode % MEAN_REWARDS == 0:
			
			avg = np.array(avg_reward)
			std = 2*np.array(std_reward)

			# plot rewards
			plt.subplot(1, 2, 2)
			plt.gca().clear()
			plt.gca().set_box_aspect(.5)
			plt.title('Reforço por episódios')
			plt.plot(avg, 'r', alpha=1.0)
			plt.fill_between(np.linspace(0, dqn[0].episode, len(avg)), avg-std, avg+std, alpha=0.3)
			plt.xlabel('Episódios')
			plt.ylabel('Reforço')

			plt.show()
			plt.pause(.1)
		
			print('\rEps: %d, Avg: %.2f, eps:%.1f' % (dqn[0].episode, dqn[0].avg_rewards[-1], dqn[0].eps))
		print('\rEps: %d, Avg: %.2f' % (dqn[0].episode, dqn[0].avg_rewards[-1]), end="")

	plt.ioff()
