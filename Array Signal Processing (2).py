# 4 anten, 2 kaynak. #ilk baş 1 kaynak sinyalini oluşturalım.
import DoA
import numpy as np
import matplotlib.pyplot as plt

pi = np.pi

t = np.linspace(0,1,1000)
N = len(t) #sample
M = 4 #anten sayısı
D = 1 #kaynak sayısı
lamda = 1
delta = lamda/2 #antenler arasındaki mesafe dalga boyunun yarısı kadar olsun.
tetha = np.array([45])
fc = 20#100 kHz
f1 = 10
original_source_signal = [20*np.cos(2*pi*fc*t)] #s(t)
radyan = pi/180
noise_db = 4

rec_sig,x_t = DoA.create_signal(M,D,lamda,delta,tetha,original_source_signal,t,noise_db)

plt.figure(figsize=(10,4))
for item in original_source_signal:
    plt.plot(t,item)
    
plt.figure(figsize=(10,4))
for item in x_t:
    plt.plot(t,item)    
plt.title("Antenler tarafından alınan sinyaller")

plt.figure(figsize=(10,4))
for item in rec_sig:
    plt.plot(t,item)
plt.title("Antenler tarafından alınan sinyaller")
plt.xlabel("Zaman(saniye)")
plt.ylabel("Genlik(V)")