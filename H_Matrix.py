import numpy as np

class H_Simulation(object):

	def __init__(self):
		#np.random.seed(2)

		self.Bandwith = 10*10**6 #bw = 10MHz
		self.N0W = self.Bandwith*10**(-174.0/10) #Noise = -174 dBm/Hz

		self.Icue_thr = 1e3*self.N0W
		self.Samples = 1 
		self.Pmax = 43. #Max power Bs
		self.Size = 200.0 #Area radius BS
		self.D2D_dist = 30 #Area rx_tx
		self.Devices = 20 #Number devices D2D
		self.Pl_const = 34.53 #urban micro lognormal
		self.Pl_alpha = 38.   #urban micro pathloss


	def loc_devices(self,Si,Di,De):
		
		def rand(Si,Di):
			#Circle form:
			temp1 = np.random.uniform(-Si/2,Si/2)
			temp  = np.sqrt((Si/2)**2-temp1**2)
			temp2 = np.random.uniform(-temp,temp)
			#Circle distribution form:
			tempL = temp1 - Di/2
			tempH = temp1 + Di/2
			if tempL<-Si/2: tempL=-Si/2
			if tempH> Si/2: tempH= Si/2
			temp3 = np.random.uniform(tempL,tempH)						
			tempL = temp2 - np.sqrt((Di/2)**2-(temp3-temp1)**2)
			tempH = temp2 + np.sqrt((Di/2)**2-(temp3-temp1)**2)
			if tempL<-Si/2: tempL=-Si/2
			if tempH> Si/2: tempH= Si/2
			temp4 = np.random.uniform(tempL,tempH)

			return [temp1,temp2],[temp3,temp4]

		rx = np.zeros((De+1,2))
		tx = np.zeros((De+1,2))

		for i in range(0,De): rx[i],tx[i]=rand(Si,Di)

		tx[De],tx[De]=rand(Si,Di)
		return rx,tx

	def channel_gen(self) :
		C_gain = []
		C_fading = []
		De = self.Devices +1
		for i in range(self.Samples):
			rx,tx = self.loc_devices(self.Size,self.D2D_dist,self.Devices)
			#Distance
			Distance = np.linalg.norm(rx.reshape(De,1,2)-tx,axis=2)
			#Path loss
			pl_ch_gain_db =  -self.Pl_const - self.Pl_alpha * np.log10(Distance)
			pl_ch_gain = 10**(pl_ch_gain_db / 10)
			multi_fading = np.random.randn(De,De) ** 2 # 0.5 x 2
			print(pl_ch_gain*multi_fading)
			fn_ch = np.maximum(pl_ch_gain*multi_fading, np.exp(-30))
			C_gain.append(fn_ch)
			C_fading.append(pl_ch_gain)

		return np.array(C_gain),np.array(C_fading)







#try:
if __name__ == '__main__':
	H_Matrix = H_Simulation()
	print(np.shape(np.array(H_Matrix.channel_gen()).reshape(2,21,21)))



#except:
#	pass