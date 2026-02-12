#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('./src')
import class_env as ce
import os
os.makedirs("./fields", exist_ok=True)
os.makedirs("./models", exist_ok=True)
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

MEAN_REWARDS = 200
plt.rcParams['figure.figsize'] = (16,8)

########################################################################
# Criando a classe para o TD learning.
class TDlearning(object):
	_counter = 0
	
	def __init__(self, parameters):
		
		# conta as instancias
		self.id = TDlearning._counter
		TDlearning._counter += 1

		self.parameters = parameters

		# metodo
		self.method = parameters['method']

		# numero de episodios
		self.episode = 0
		
		# cria o ambiente
		self.env = ce.Env(parameters)
		
		# tamanho dos espacos de estados e acoes
		self.num_states = np.prod(np.array(self.env.num_states))
		self.num_actions = self.env.action_space.n

		# parametros de aprendizado
		self.gamma = parameters['gamma']
		self.eps = parameters['eps']
		self.alpha = parameters['alpha']

		# log file (name depends on the method)
		self.logfile = parameters['q-file'] + f'[{self.id}].npy'
		if self.method == 'Sarsa':
			self.logfile = 'sarsa_' + self.logfile
		elif self.method == 'Qlearning':
			self.logfile = 'qlearning_' + self.logfile
		else: print("Não salvou...")
		self.logfile = 'models/' + self.logfile
		
		# reseta a politica
		self.reset()

	##########################################
	# reseta a funcao acao-valor
	def reset(self):
		
		# reseta o ambiente
		S, _ = self.env.reset()
		
		# Q(s,a)
		self.Q = np.zeros((self.num_states, self.num_actions))
		
		# rewards
		self.rewards = []
		self.avg_rewards = []

		# carrega tabela pre-computada se for o caso
		if self.parameters['load_Q']:
			self.load()
		
	##########################################
	# retorna a politica corrente
	def curr_policy(self, copy=False):
		if copy:
			return partial(self.TabularEpsilonGreedyPolicy, np.copy(self.Q))
		else:
			return partial(self.TabularEpsilonGreedyPolicy, self.Q)
		
	########################################
	# salva tabela Q(s,a)
	def save(self):
		with open(self.logfile, 'wb') as f:
			np.savez(f, Q=self.Q, episodes=self.episode, rewards=self.rewards)

	########################################
	# carrega tabela Q(s,a)
	def load(self):
		with open(self.logfile, 'rb') as f:
			data = np.load(f)
			self.Q = data['Q']
			self.episode = data['episodes']
			self.rewards = list(data['rewards'])
			self.avg_rewards = []
			for i in range(1,len(self.rewards)):
				a = self.rewards[:i]
				self.avg_rewards.append(np.mean(a[-MEAN_REWARDS:]))
		
	##########################################
	def close(self):
		self.env.close()

	##########################################
	# escolha da açao (epsilon-soft)
	def TabularEpsilonGreedyPolicy(self, Q, state):

		# numero total de acoes
		nactions = Q.shape[1]
		
		if np.random.random() < self.eps:
			return np.random.choice(nactions)
		else:
			return Q[state, :].argmax()

	##########################################
	def sarsa(self, S, A):

		# passo de interacao com o ambiente
		Sl, R, done, _, _ = self.env.step(A)
		
		# escolhe A' a partir de S'
		Al = self.policy(Sl)
		
		# update de Q(s,a)
		self.Q[S, A] = self.Q[S, A] + self.alpha*(R + self.gamma*self.Q[Sl, Al] - self.Q[S, A])
		
		return Sl, Al, R, done

	##########################################
	def qlearning(self, S):
		
		# \pi(s)
		A = self.policy(S)

		# passo de interacao com o ambiente
		Sl, R, done, _, _ = self.env.step(A)
		
		self.Q[S, A] = self.Q[S, A] + self.alpha*(R + self.gamma*self.Q[Sl, :].max() - self.Q[S, A])
		
		return Sl, R, done
		
	##########################################
	# simula um episodio até o fim seguindo a politica corente
	def rollout(self):
		
		# inicia o ambiente (começa aleatoriamente)
		S, _ = self.env.reset()
		
		# \pi(s)
		A = self.policy(S)

		# lista de rewards
		rewards = []

		while True:
			if self.method == 'Sarsa':
				Sl, Al, R, done = self.sarsa(S, A)
				# proximo estado e ação
				S = Sl
				A = Al
				
			elif self.method == 'Qlearning':
				Sl, R, done = self.qlearning(S)
				# proximo estado
				S = Sl

			# Salva rewards
			rewards.append(R)

			# chegou a um estado terminal?
			if done:
				break

		return rewards
		
	##########################################
	# Executando um episódio.
	def runEpisode(self):

		# novo episodio
		self.episode += 1

		# pega a politica corrente (on-policy)
		self.policy = self.curr_policy()

		# gera um episodio seguindo a politica corrente
		rewards = self.rollout()
		
		# renderiza o ambiente
		if (self.episode % MEAN_REWARDS) == 0:
			# render
			if self.id == 0:
				plt.subplot(1, 2, 1)
				plt.gca().clear()
				self.env.render(self.Q)
			
			# salva a tabela Q
			if self.parameters['save_Q']:
				self.save()

		# rewards
		self.rewards.append(np.sum(rewards))
		self.avg_rewards.append(np.mean(self.rewards[-MEAN_REWARDS:]))

########################################################################
# Código principal:
# - episodes: número de episódios
# - gamma: fator de desconto
# - eps: $\varepsilon$
# - alpha: $\alpha$
# - method: *Sarsa* ou *Q-learning*
# - save_Q: salva tabela *Q*
# - load_Q: carrega tabela *Q*
# - q-file: arquivo da tabela *Q*
if __name__ == '__main__':
	
	plt.ion()
	
	# parametros
	parameters = {'episodes'  : 20000,
				  'gamma'     : 0.99,
				  'eps'       : 0.1,
				  'alpha'     : 0.5,
				  'method'    : 'Sarsa', #'Sarsa' ou 'Qlearning'
				  'save_Q'    : True,
				  'load_Q'    : False,
				  'q-file'    : 'qtable',
				  'map'		  : 'imgs/islands.png',
				  'xgoal'     : np.array([110.0, 90.0]),
				  'nagents'	  : 2,
				  'xlim'	  : np.array([0.0, 120.0]),
				  'ylim'	  : np.array([0.0, 100.0]),
				  'resolution': 500,
				  }

	# TD algorithm
	td = [TDlearning(parameters) for _ in range(parameters['nagents'])]
	
	avg_reward = []
	std_reward = []
	
	while td[0].episode <= parameters['episodes']:
		
		# roda um episodio
		for t in td:
			t.runEpisode()
			
		avg_reward.append(np.mean([t.rewards[-1] for t in td]))
		std_reward.append(np.std([t.rewards[-1] for t in td]))
		
		# print e plot
		if td[0].episode % MEAN_REWARDS == 0:

			avg = np.array(avg_reward)
			std = 2*np.array(std_reward)
			
			# plot rewards
			plt.subplot(1, 2, 2)
			plt.gca().clear()
			plt.gca().set_box_aspect(.5)
			plt.title('Reforço por episódios')
			plt.plot(avg, 'r', alpha=1.0)
			plt.fill_between(np.linspace(0, td[0].episode, len(avg)), avg-std, avg+std, alpha=0.3)
			plt.xlabel('Episódios')
			plt.ylabel('Reforço')

			plt.show()
			plt.pause(.1)
		
			print('\rEps: %d, Avg: %.2f' % (td[0].episode, td[0].avg_rewards[-1]))
		print('\rEps: %d, Avg: %.2f' % (td[0].episode, td[0].avg_rewards[-1]), end="")

	plt.ioff()
	[t.close() for t in td]
