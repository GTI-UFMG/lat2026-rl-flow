# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

########################################
# classe do fluxo
########################################
class Flow:
	
	_counter = 0
	
	########################################
	# construtor
	########################################
	def __init__(self, parameters, LV, dim = 2):
		
		# conta as instancias
		self.id = Flow._counter
		Flow._counter += 1
		
		# dimensao do espaco de trabalho
		self.dim = dim
		
		# seta o goal
		self.goal = np.array([np.squeeze(parameters['xgoal'])])
		
		# is time-varying
		self.timevarying = False
		
		# cor
		self.color = 'k'
		
		# seta parametros
		self.setPar(LV)
	
	########################################
	# set parameters
	########################################
	def setPar(self, LV):
		
		'''rz, delta, gamma, omega = LV'''
		#self.p, self.delta, self.gamma, self.omega = LV
		self.p, self.delta, self.gamma = LV
		
		'''Quadrant = [[xinf,xsup],[yinf,ysup], goal_line, max_flow, omega]'''
		#self.xlim, self.ylim, self.goal_line, self.max_flow, self.omega = LV
		
		# going with flows
		#self.p = LV
		
		self.kappa = 1.0
		self.alpha = 2.0
		
	########################################
	# calcula equacao do fluxo
	########################################
	def currentField(self, q, t):
		
		###################################
		rz = self.p
		delta = self.delta
		Gamma = self.gamma
		if self.timevarying:
			Gamma *= np.cos(self.omega*t)
		nr = np.linalg.norm(q - self.p, axis=1)
		Vx = -Gamma * ((q[:,1] - self.p[1])/(2.*np.pi*nr**2.)) * (1.0 - np.exp(-(nr**2./delta**2.0)))
		Vy =  Gamma * ((q[:,0] - self.p[0])/(2.*np.pi*nr**2.)) * (1.0 - np.exp(-(nr**2./delta**2.0)))
		
		###################################
		# "going with flows" model
		'''s = 1.0
		A = 0.02
		# se varia no tempo
		if self.timevarying:
			A *= np.cos(.02*t)
			
		X = q[:,0] - self.p[0]
		Y = q[:,1] - self.p[1]
		Vx = -np.pi*A*np.sin(np.pi*X/s)*np.cos(np.pi*Y/s)
		Vy =  np.pi*A*np.cos(np.pi*X/s)*np.sin(np.pi*Y/s)'''
		
		'''###################################
		X = q[:,0]
		Y = q[:,1]
		
		deduction = np.abs(Y - self.goal_line)
		normalized_deduction = deduction/np.max(deduction) * self.max_flow
		
		Vx = (self.max_flow - normalized_deduction) * np.cos(self.omega*t)
		Vy = Y * 0.0
		Vx[(X < self.xlim[0]) | (X >= self.xlim[1])] = 0.0
		Vx[(Y <= self.ylim[0]) | (Y > self.ylim[1])] = 0.0'''
		
		###################################
		# retorna o fluxo
		V = np.column_stack((Vx, Vy));
		
		return V
		
	########################################
	# desenha cada no sensor
	########################################
	def draw(self):
		try:
			plt.lot(self.p[0], self.p[1], 'o', color = self.color, markersize = 3, markeredgewidth = 2)
		except:
			None
		
	########################################
	def close(self):
		None
