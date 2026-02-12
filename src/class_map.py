# -*- coding: utf-8 -*-
# Introdução ao Aprendizado por Reforço - PPGEE
# Prof. Armando Alves Neto
########################################################################
import matplotlib.pyplot as plt
import numpy as np
import cv2

########################################
# classe do mapa
########################################
class Map:
	########################################
	# construtor
	def __init__(	self, parameters, img='imgs/void.png'):
		
		# seta limites do ambiente
		self.xlim, self.ylim = [parameters.get(k) for k in ['xlim', 'ylim']]

		# cria mapa
		self.init2D(img)
		
		
	########################################
	# ambientes em 2D
	def init2D(self, image):

		# le a imagem
		I = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

		# linhas e colunas da imagem
		self.nrow = I.shape[0]
		self.ncol = I.shape[1]

		# binariza imagem
		(thresh, I) = cv2.threshold(I, 127, 255, cv2.THRESH_BINARY)

		# inverte a imagem em y
		self.mapa = cv2.flip(I, 0)

		# parametros de conversao
		self.mx = float(self.ncol) / float(self.xlim[1] - self.xlim[0])
		self.my = float(self.nrow) / float(self.ylim[1] - self.ylim[0])

	########################################
	# verifica colisao com os obstaculos
	def collision(self, q):

		# posicao de colisao na imagem
		px, py = self.mts2px(q)
		col = int(px)
		lin = int(py)

		# verifica se esta dentro do ambiente
		if (lin <= 0) or (lin >= self.nrow):
			return True
		if (col <= 0) or (col >= self.ncol):
			return True

		# colisao
		try:
			if self.mapa.item(lin, col) < 127:
				return True
		except IndexError:
			None

		return False

	########################################
	# transforma pontos no mundo real para pixels na imagem
	def mts2px(self, q):
		qx, qy = q
		# conversao
		px = (qx - self.xlim[0])*self.mx
		py = self.nrow - (qy - self.ylim[0])*self.my

		return px, py

	########################################
	# desenha a imagem distorcida em metros
	def draw(self):
		
		# plota mapa real e o mapa observado
		plt.imshow(self.mapa, cmap='gray', extent=[self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]], alpha=0.5)

	########################################
	def __del__(self):
		None
