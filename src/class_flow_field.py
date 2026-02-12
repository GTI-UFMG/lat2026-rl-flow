# -*- coding: utf-8 -*-
import class_flow
import matplotlib.pyplot as plt
import numpy as np
import progressbar
	
VMAX = 3.0

########################################
# classe do campo dos nos sensores
########################################
class Field:
	########################################
	# construtor
	########################################
	def __init__(self, parameters, LV, dim = 2):
		
		# dimensao do espaco de trabalho
		self.dim = dim
		
		# tipo da funcao de intensidade
		self.type = type
		
		# seta limites do ambiente
		self.xlim, self.ylim = [parameters.get(k) for k in ['xlim', 'ylim']]
		
		# resolucao do fluxo
		self.resolution = parameters['resolution']
		
		# otimiza energy ou tempo?
		self.energy = True
		
		# inicializacao depende da dimensao
		if self.dim == 2:
			self.init2D(parameters, LV)
			
		# pega maximo valor do flow
		self.Vfm = self.getMaxFlowValue(t=0.0)
		print('Max flow value = %.4f' % self.Vfm)
		
		# parametros do fluxo
		self.kappa = self.flows[0].kappa
		self.alpha = self.flows[0].alpha
		
	########################################
	# ambientes em 2D
	########################################
	def init2D(self, parameters, LV = [], gamma = []):
		
		# cria fluxos em posições especificas
		self.flows = []
		for lv in LV:
			self.flows.append(class_flow.Flow(parameters, lv))
		
		#####################
		# carrega um valor previamente calculado
		xp = [lv[0][0] for lv in LV]
		yp = [lv[0][1] for lv in LV]
		self.fieldfile = 'fields/field' + '_%.4f' % sum(xp) + '_%.4f' % sum(yp) + '_%d' % self.resolution + '.npy'
		try:
			with open(self.fieldfile, 'rb') as f:
				#print('Loading %s...' % self.fieldfile)
				self.Z = np.load(f)
		except:
			print('File %s not found...' % self.fieldfile)
			# monta mapa do sensor field
			self.get_value()
			with open(self.fieldfile, 'wb') as f:
				np.save(f, self.Z)
				
		# parametros de conversao
		self.mx = float(self.resolution) / float(self.xlim[1] - self.xlim[0])
		self.my = float(self.resolution) / float(self.ylim[1] - self.ylim[0])
	
	########################################
	# sensor field intensity function
	########################################
	def Intensity(self, p, vnet, dt, t):
		
		# calcula velocidade do fluxo
		#vf = np.squeeze(self.getFlow(np.array([p])))
		i, j = self.mts2px(p)
		vf = self.Z[i][j]
		
		#############################################
		# minimiza energia
		if self.energy:
			vnet = self.Vfm*(vnet/np.linalg.norm(vnet))
			vstill = vnet - vf
				
			return self.kappa*(np.linalg.norm(vstill)**self.alpha)*dt
		
		#############################################
		# minimiza o tempo
		else:		
			vstill = vf - vmax
			return np.linalg.norm(vstill)
			
	########################################
	# sensor field intensity function
	########################################
	def getMaxFlowValue(self, t=0.0):
			
		m = 100
		XX, YY = np.meshgrid(np.linspace(self.xlim[0], self.xlim[1], m), np.linspace(self.ylim[0], self.ylim[1], m))
		XY = np.dstack([XX, YY]).reshape(-1, 2)
		
		# para vários pontos no mundo, calcula o valor do fluxo
		Vf = self.getFlow(XY, t)
		Vf = np.linalg.norm(Vf, axis=1)
		return Vf.max()
	
	########################################
	# get flow vector
	########################################
	def getFlow(self, p, t=0.0):
		
		# calcula velocidade do fluxo
		for f in self.flows:
			try:
				Vf += f.currentField(p, t)
			except:
				Vf = f.currentField(p, t)
		return np.squeeze(Vf)
		
	########################################
	# function to draw the map
	########################################
	def get_value(self):
		
		bar = progressbar.ProgressBar(self.resolution)
		bar.start()
		
		x = np.linspace(self.xlim[0], self.xlim[1], self.resolution)
		y = np.linspace(self.ylim[0], self.ylim[1], self.resolution)
		
		# mapa vazio
		self.Z = [0] * self.resolution
		for i in range(self.resolution):
			self.Z[i] = [0] * self.resolution
		
		# para todos os pontos do mapa
		for i in range(self.resolution):
			for j in range(self.resolution):
				# pega a probabilidade de detecção
				p = np.array([x[i], y[j]])
				self.Z[i][j] = np.squeeze(self.getFlow(np.array([p])))
			
			try: bar.update(i) 
			except: None
		
	########################################
	# transforma pontos no mundo real para pixels na imagem
	########################################
	def mts2px(self, q):
		
		qx, qy = q
		
		# conversao
		px = (qx - self.xlim[0])*self.mx
		py = (qy - self.ylim[0])*self.my
		px = min(self.resolution-1 , max(px, 0))
		py = min(self.resolution-1 , max(py, 0))
		
		return int(px), int(py)
		
	########################################
	# desenha a imagem distorcida em metros
	########################################
	def draw(self, t, field=True):
		
		if self.dim == 2:
			
			# Plot o field
			if field:
				m = 40
				x = np.linspace(self.xlim[0], self.xlim[1], m)
				y = np.linspace(self.ylim[0], self.ylim[1], m)
				XX, YY = np.meshgrid(x, y)
				XY = np.dstack([XX, YY]).reshape(-1, 2)

				for f in self.flows:
					try:
						V += f.currentField(XY, t)
					except:
						V = f.currentField(XY, t)
						
				Vx = V[:,0]
				Vy = V[:,1]
				M = np.hypot(Vx, Vy)
				
				obj = plt.gca().quiver(XX, YY, Vx, Vy, angles='xy', scale_units='xy', scale=0.15, alpha=.1)
				#plt.colorbar(obj, cmap='cool')
				#plt.gca().set_facecolor("cyan")
				#plt.gca().set_alpha(.3)
		
				# desenha todos os centros dos flows
				[f.draw() for f in self.flows]
			
			# coloca os labels
			try:
				FONTSIZE = 24
				plt.xlabel(r"$x$[m]", fontsize=FONTSIZE)
				plt.ylabel(r"$y$[m]", fontsize=FONTSIZE)
			except:
				plt.xlabel("x[m]")
				plt.ylabel("y[m]")
		
	########################################
	def close(self):
		None
