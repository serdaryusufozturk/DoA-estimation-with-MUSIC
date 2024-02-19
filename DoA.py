#!/usr/bin/env python
# coding: utf-8

# In[2]:


import math
import cmath
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'notebook')
pi = np.pi
radyan = np.pi/180


# In[9]:


def create_signal(M,D,lamda,delta,tetha,signal,t,target_noise_db):
    #M: Anten sayısı  D:Kaynak sayısı, lamda: Dalga boyu, delta: Antenler arası mesafe, tetha: Sinyal geliş açı matrisi
    #signal: Kaynaktan gelen sinyal, N: Örnek sayısı.
    
    mü = []
    #noise = np.random.randn(M,len(t))

    target_noise_watts = 10 ** (target_noise_db / 10)
    mean_noise = 0
    noise = np.random.normal(mean_noise, np.sqrt(target_noise_watts), (M,len(t)))

    pre = ((-2*pi)/lamda)
    for item in tetha:
        mü.append(pre*delta*np.sin(item*radyan))

#Aşağıdaki döngü bloğu ise makaledeki eq.(7) yi uygulamak içindir. 
    a_mü= np.array([[0 for row in range(M)] for col in range(D)],dtype=np.complex_) #dtype=np.complex_ yapmazsak arraye complex değerler koyamıyoruz.
    for element in range(D):# kaynak = 1
        for item in range(M):# Anten = 4
            a_mü[element][item]=(np.exp(1j*mü[element]*item)) #a(µi) vektörü.

        
    A = np.transpose(a_mü) #steering matrix
    x_t = np.matmul(A,signal)
    received_signal = x_t + noise
    
    return received_signal,x_t


# In[10]:


def music(received_signal,angle,M,D,lamda,delta):
    Cov_rec_sig = np.cov(received_signal)
    print(f"Kovaryans matris\n{Cov_rec_sig}") #Kovaryans matrisi bulduk.
    eigenvalues, eigenvectors = np.linalg.eig(Cov_rec_sig)
    print(f'\n\nEigenvalues:\n{eigenvalues}\n\nEigenvector:\n{eigenvectors}')
    eigenvalues_yedek = eigenvalues
    signal_subspace = D
    noise_subspace = M-D
    Vn = np.array([[0]*(M-D)]*(M),dtype=np.complex_)
    eigenvectorss = np.array(eigenvectors).T #eigenvectorsün transpozunu almamın sebebi sütunları çekmektense onları satır olarak çekmenin daha kolay olması.
    for item in range(M-D):
        Min_Eigenv = min(eigenvalues_yedek)
        out = np.where(Min_Eigenv == eigenvalues)#bulduğumuz min eigenvalue nin dizi içindeki indeksini buluyoruz.
        out1 = np.where(Min_Eigenv == eigenvalues_yedek) 
        #Yukarıda eigenvalue_yedek değişkeni kullandık çünkü oradan minimum değeri çekip o dizide siliyoruz. Ana dizide de bunun indeksini buluyoruz.
        indeks = out[0][0]# indeks array şeklinde verildi. Saf indeks değerini çekiyoruz. Bu indeks değeriyle eigenvectoru bulacağız.
        indeks1 = out1[0][0]# yedek veriden değeri silmek için diğer indeksi çekiyoruz
        print(f'\n\nMinimum Eigenvalue:\n{Min_Eigenv}')
        Vn[:,item] = eigenvectorss[indeks]
        eigenvalues_yedek = np.delete(eigenvalues_yedek,indeks1)
    
    
    
    #for item in range(M-D)
     #   Vn[:,item] = eigenvectorss[3]
      #  Vn[:,1] = eigenvectorss[2]
       # Vn[:,2] = eigenvectorss[1]
    
    
    #Vn = np.array([[[0]*M]*(M-D)])
    #Vn[0] = np.array([eigenvectors[:,3]])
    #Vn[1] = np.array([eigenvectors[:,2]])
    #Vn[2] = np.array([eigenvectors[:,1]])
    print(f'\n\nVn:\n{Vn}')
   # for i in range(M-D):
   #     out = np.where(Min_Eigenvalues[i] == eigenvalues) #bulduğumuz min eigenvalue nin dizi içindeki indeksini buluyoruz.
   #     indeks = out[0][0]# indeks array şeklinde verildi. Saf indeks değerini çekiyoruz. Bu indeks değeriyle eigenvectoru bulacağız.
   #     print(f'\n\nMinimum Eigenvalue:\n{Min_Eigenv}')
   #     Vn[:,i] = np.array([eigenvectors[:,indeks]]).T #Vn in transpozunu alıp dikey vektör haline getirdim. Eigenvectorler genelde sutun olarak alınır.
   #     print(f'\n\nVn:\n{Vn}')
    Vn_H = np.conjugate(Vn.T)
    print(f'\n\nHermitian Vn:\n{Vn_H}')
    pre = ((-2*np.pi)/lamda)
    mü_new = []
    for item in angle:
        mü_new.append(pre*delta*np.sin(item*radyan))

    a_mü_new = np.array([[0 for row in range(len(angle))] for col in range(M)],dtype=np.complex_) #dtype=np.complex_ yapmazsak arraye complex değerler koyamıyoruz.
    for element in range(len(angle)):
        for item in range(M):
            a_mü_new[item][element]=(np.exp(1j*mü_new[element]*item)) #a(µi) vektörü.
    a_mü_new.shape #4x1000
    a_mü_new_H = np.conjugate(a_mü_new.T)
    
    a_mü_new_H.shape #1000x4

    Pmusic = np.array([0 for i in range(len(angle))],dtype=np.complex_)
    for item in range(len(angle)):
        if item !=len(angle):
            final = np.matmul(a_mü_new_H[item:item+1],Vn @ Vn_H @ a_mü_new[:,item])
        else:
            final = np.matmul(a_mü_new_H[item:],Vn @ Vn_H @ a_mü_new[:,item])
        
    #first = np.matmul(a_mü_new_H[item],Vn)
    #second = np.matmul(first,Vn_H)
    #final = np.matmul(second,a_mü_new[:,item])
        Pmusic[item]=(1/(final))
    Pdb = np.array([0]*len(angle))
    for item in range(len(angle)):
        Pdb[item] = 10*math.log(abs(Pmusic[item]))
    return Pdb


# In[11]:


def create_signal_UCA(M,D,lamda,delta,phi,signal,t,r,target_noise_db):
    #M: Anten sayısı  D:Kaynak sayısı, lamda: Dalga boyu, delta: Antenler arası mesafe, tetha: Sinyal geliş açı matrisi
    #signal: Kaynaktan gelen sinyal, N: Örnek sayısı.
    pi = np.pi
    radyan = pi/180
    #noise = 10*np.random.randn(M,len(t))
    r = lamda/(4*np.sin(np.pi/M))
    target_noise_watts = 10 ** (target_noise_db / 10)
    mean_noise = 0
    noise = np.random.normal(mean_noise, np.sqrt(target_noise_watts), (M,len(t)))
   # x_n = np.array([0 for i in range(M)])

    #x_n = np.array([r*np.cos(2*pi*i/M) for i in range(M)])
    #print(f'x_n: {x_n}\n')
    #y_n = np.array([r*np.sin(2*pi*i/M) for i in range(M)])
    #print(f'y_n: {y_n}\n')
    #p_n = np.array([0 for i in range(M)])
    #for item in phi:
     #   for n in range(M):
      #      p_n[n] = x_n[n]*np.cos(item*radyan)+y_n[n]*np.sin(item*radyan) #Açı derece cinsinden yazıldığı için radyana çevirdik.
           # p_n[n] = n
       #     print(p_n[n])
    #print(p_n)
    a_phi = np.array([[0]*D]*M,dtype=np.complex_)
    
    #Alttaki döngü array manifoldu oluşturuyor.
    for item in range(D):#sinyal sayısı
        for n in range(M):#anten sayısı
            #a_phi[n][item] = np.exp(1j*beta*r*np.cos(phi-2*n*pi/M))
            print(phi[item])
            a_phi[n][item] = np.exp(1j*2*pi*(r/lamda)*np.cos(np.deg2rad(phi[item])-((n/M)*2*pi))) #exp in içine bak
    
    print(a_phi)
    print(f'a_phi dimensions: {a_phi.shape}')
    x_k = (a_phi @ signal)  #X(k) nın gürültüsüz halinin oluşturulması.
    print(x_k)
    print(x_k.shape)
    received_signal = x_k + noise
    print(received_signal.shape)
    
    return received_signal,x_k


# In[12]:


def music_UCA(received_signal,angle,M,D,lamda,delta):
    pi = np.pi
    radyan = np.pi/180
    r = lamda/(4*np.sin(np.pi/M))
    Cov_rec_sig = np.cov(received_signal)
    print(f"Kovaryans matris\n{Cov_rec_sig}") #Kovaryans matrisi bulduk.
    eigenvalues, eigenvectors = np.linalg.eig(Cov_rec_sig)
    print(f'\n\nEigenvalues:\n{eigenvalues}\n\nEigenvector:\n{eigenvectors}')
    eigenvalues_yedek = eigenvalues
    signal_subspace = D
    noise_subspace = M-D
    Vn = np.array([[0]*(M-D)]*(M),dtype=np.complex_)
    eigenvectorss = np.array(eigenvectors).T #eigenvectorsün transpozunu almamın sebebi sütunları çekmektense onları satır olarak çekmenin daha kolay olması.
    for item in range(M-D):
        Min_Eigenv = min(eigenvalues_yedek)
        out = np.where(Min_Eigenv == eigenvalues)#bulduğumuz min eigenvalue nin dizi içindeki indeksini buluyoruz.
        out1 = np.where(Min_Eigenv == eigenvalues_yedek) 
        #Yukarıda eigenvalue_yedek değişkeni kullandık çünkü oradan minimum değeri çekip o dizide siliyoruz. Ana dizide de bunun indeksini buluyoruz.
        indeks = out[0][0]# indeks array şeklinde verildi. Saf indeks değerini çekiyoruz. Bu indeks değeriyle eigenvectoru bulacağız.
        indeks1 = out1[0][0]# yedek veriden değeri silmek için diğer indeksi çekiyoruz
        print(f'\n\nMinimum Eigenvalue:\n{Min_Eigenv}')
        Vn[:,item] = eigenvectorss[indeks]
        eigenvalues_yedek = np.delete(eigenvalues_yedek,indeks1)
    
    print(f'\n\nVn:\n{Vn}')
 
    Vn_H = np.conjugate(Vn.T)
    print(f'\n\nHermitian Vn:\n{Vn_H}')
    
    a_phi = np.array([[0]*len(angle)]*M,dtype=np.complex_)
    
    #Alttaki döngü array manifoldu oluşturuyor.
    for item in range(len(angle)):#sinyal sayısı
        for n in range(M):#anten sayısı
            #a_phi[n][item] = np.exp(1j*beta*r*np.cos(phi-2*n*pi/M))
            a_phi[n][item] = np.exp(1j*2*pi*(r/lamda)*np.cos(np.deg2rad(angle[item])-((n/M)*2*np.pi))) #exp in içine bak
    a_phi_H = np.conjugate(a_phi.T)
    print(f'dimension of a_phi: {a_phi.shape}')
    print(f'dimension of a a_phi_H: {a_phi_H.shape}')
    
    Pmusic = np.array([0 for i in range(len(angle))],dtype=np.complex_)
    for item in range(len(angle)):
        if item !=len(angle):
            final = np.matmul(a_phi_H[item:item+1],Vn @ Vn_H @ a_phi[:,item])
        else:
            final = np.matmul(a_phi_H[item:],Vn @ Vn_H @ a_phi[:,item])
        
        Pmusic[item]=(1/(final))
    for item in range(len(angle)):
        Pmusic[item] = 10*math.log(abs(Pmusic[item]))
    return Pmusic


# In[13]:


def create_signal_UCA_e(M,D,lamda,delta,phi,signal,t,elevation,target_noise_db):
    #M: Anten sayısı  D:Kaynak sayısı, lamda: Dalga boyu, delta: Antenler arası mesafe, tetha: Sinyal geliş açı matrisi
    #signal: Kaynaktan gelen sinyal, N: Örnek sayısı.
    pi = np.pi
    radyan = pi/180
    
    target_noise_watts = 10 ** (target_noise_db / 10)
    mean_noise = 0
    noise = np.random.normal(mean_noise, np.sqrt(target_noise_watts), (M,len(t)))
    
    r = lamda/(4*np.sin(np.pi/M))
   # x_n = np.array([0 for i in range(M)])

    #x_n = np.array([r*np.cos(2*pi*i/M) for i in range(M)])
    #print(f'x_n: {x_n}\n')
    #y_n = np.array([r*np.sin(2*pi*i/M) for i in range(M)])
    #print(f'y_n: {y_n}\n')
    #p_n = np.array([0 for i in range(M)])
    #for item in phi:
     #   for n in range(M):
      #      p_n[n] = x_n[n]*np.cos(item*radyan)+y_n[n]*np.sin(item*radyan) #Açı derece cinsinden yazıldığı için radyana çevirdik.
           # p_n[n] = n
       #     print(p_n[n])
    #print(p_n)
    a_phi = np.array([[0]*D]*M,dtype=np.complex_)
    
    #Alttaki döngü array manifoldu oluşturuyor.
    for item in range(D):#sinyal sayısı
        for n in range(M):#anten sayısı
            #a_phi[n][item] = np.exp(1j*beta*r*np.cos(phi-2*n*pi/M))
            print(phi[item])
            a_phi[n][item] = np.exp(1j*2*pi*(r/lamda)*np.sin(np.deg2rad(elevation[item]))*np.cos(np.deg2rad(phi[item])-((n/M)*2*pi))) #exp in içine bak
    
    print(a_phi)
    print(f'a_phi dimensions: {a_phi.shape}')
    x_k = (a_phi @ signal)  #X(k) nın gürültüsüz halinin oluşturulması.
    print(x_k)
    print(x_k.shape)
    received_signal = x_k + noise
    print(received_signal.shape)
    
    return received_signal,x_k


# In[14]:


def music_UCA_2D(received_signal,angle,M,D,lamda,delta,elevation):
    pi = np.pi
    radyan = np.pi/180
    r = lamda/(4*np.sin(np.pi/M))
    a_phi = np.array([[[0 for col in range(len(angle))]for row in range(M)] for x in range(len(elevation))],dtype=np.complex_)
     #Alttaki döngü array manifoldu oluşturuyor.
    for item in range(len(elevation)):#sinyal sayısı
        for element in range(M):
            for n in range(len(angle)):#anten sayısı
                #a_phi[n][item] = np.exp(1j*beta*r*np.cos(phi-2*n*pi/M))
                a_phi[item][element][n] = np.exp(1j*2*pi*(r/lamda)*np.sin(np.deg2rad(elevation[item]))*np.cos(np.deg2rad(angle[n])-((element/M)*2*np.pi))) #exp in içine bak
    print(f'a_phi: \n {a_phi}')
    #Alttaki satırda a_phi nin hermisyenini elevation için teker teker girip 2 boyutta oluşturuyoruz.
    a_phi_H = np.array([np.conjugate(a_phi[item].T) for item in range(len(elevation))])
    print(f'dimension of a_phi: {a_phi.shape}')
    print(f'dimension of a a_phi_H: {a_phi_H.shape}')
    
    
    Cov_rec_sig = np.cov(received_signal)
    print(f"Kovaryans matris\n{Cov_rec_sig}") #Kovaryans matrisi bulduk.
    eigenvalues, eigenvectors = np.linalg.eig(Cov_rec_sig)
    print(f'\n\nEigenvalues:\n{eigenvalues}\n\nEigenvector:\n{eigenvectors}')
    eigenvalues_yedek = eigenvalues
    signal_subspace = D
    noise_subspace = M-D
    Vn = np.array([[0]*(M-D)]*(M),dtype=np.complex_)
    eigenvectorss = np.array(eigenvectors).T #eigenvectorsün transpozunu almamın sebebi sütunları çekmektense onları satır olarak çekmenin daha kolay olması.
    for item in range(M-D):
        Min_Eigenv = min(eigenvalues_yedek)
        out = np.where(Min_Eigenv == eigenvalues)#bulduğumuz min eigenvalue nin dizi içindeki indeksini buluyoruz.
        out1 = np.where(Min_Eigenv == eigenvalues_yedek) 
        #Yukarıda eigenvalue_yedek değişkeni kullandık çünkü oradan minimum değeri çekip o dizide siliyoruz. Ana dizide de bunun indeksini buluyoruz.
        indeks = out[0][0]# indeks array şeklinde verildi. Saf indeks değerini çekiyoruz. Bu indeks değeriyle eigenvectoru bulacağız.
        indeks1 = out1[0][0]# yedek veriden değeri silmek için diğer indeksi çekiyoruz
        print(f'\n\nMinimum Eigenvalue:\n{Min_Eigenv}')
        Vn[:,item] = eigenvectorss[indeks]
        eigenvalues_yedek = np.delete(eigenvalues_yedek,indeks1)
    
    print(f'\n\nVn:\n{Vn}')
 
    Vn_H = np.conjugate(Vn.T)
    print(f'\n\nHermitian Vn:\n{Vn_H}')
    
    
    
   
    
    Pmusic = np.array([[0 for i in range(len(angle))]for row in range(len(elevation))],dtype=np.complex_)
    for element in range(len(elevation)):
        for item in range(len(angle)):
            if item !=360:
                final = np.matmul(a_phi_H[element][item:item+1],Vn @ Vn_H @ a_phi[element][:,item])
            else:
                final = np.matmul(a_phi_H[element][item:],Vn @ Vn_H @ a_phi[element][:,item])
        
            Pmusic[element][item]=10*math.log(abs(1/(final)))
    print(f'Pmusic:\n{Pmusic}')
    return Pmusic


# In[15]:


import numpy as np
def closest(lst, K):
     
    lst = np.asarray(lst)
    idx = (np.abs(lst - K)).argmin()
    return lst[idx]


# In[16]:


def root_MUSIC(received_signal,M,D,lamda,delta):
    Cov_rec_sig = np.cov(received_signal)
    print(f"Kovaryans matris\n{Cov_rec_sig}") #Kovaryans matrisi bulduk.
    eigenvalues, eigenvectors = np.linalg.eig(Cov_rec_sig)
    print(f'\n\nEigenvalues:\n{eigenvalues}\n\nEigenvector:\n{eigenvectors}')
    eigenvalues_yedek = eigenvalues
    signal_subspace = D
    noise_subspace = M-D
    Vn = np.array([[0]*(M-D)]*(M),dtype=np.complex_)
    eigenvectorss = np.array(eigenvectors).T #eigenvectorsün transpozunu almamın sebebi sütunları çekmektense onları satır olarak çekmenin daha kolay olması.
    for item in range(M-D):
        Min_Eigenv = min(eigenvalues_yedek)
        out = np.where(Min_Eigenv == eigenvalues)#bulduğumuz min eigenvalue nin dizi içindeki indeksini buluyoruz.
        out1 = np.where(Min_Eigenv == eigenvalues_yedek) 
        #Yukarıda eigenvalue_yedek değişkeni kullandık çünkü oradan minimum değeri çekip o dizide siliyoruz. Ana dizide de bunun indeksini buluyoruz.
        indeks = out[0][0]# indeks array şeklinde verildi. Saf indeks değerini çekiyoruz. Bu indeks değeriyle eigenvectoru bulacağız.
        indeks1 = out1[0][0]# yedek veriden değeri silmek için diğer indeksi çekiyoruz
        print(f'\n\nMinimum Eigenvalue:\n{Min_Eigenv}')
        Vn[:,item] = eigenvectorss[indeks]
        eigenvalues_yedek = np.delete(eigenvalues_yedek,indeks1)
    

    print(f'\n\nVn:\n{Vn}')
    Vn_H = np.conjugate(Vn.T)
    print(f'\n\nHermitian Vn:\n{Vn_H}')
    
    A = Vn @ Vn_H #MxM lik bir matris gelecek. M anten sayısıdır.
    l = np.arange((-M+1),(M),1) #M-1 yazmadık çünkü arange fonksiyonu son değeri dahil etmiyor.
    
    a_l = np.array([sum(np.diag(A,k=item)) for item in l]) #Matrisin her köşegenini toplayıp a_l vektörüne atar.
    genlik = np.array([0 for i in range(len(l)-1)],dtype=np.float_) #len(l)-1 olmasının sebebi l-1 kadar kök gelmesinden kaynaklı.
    #Coef = np.flip(a_l) #katsayıları en yüksek dereceden en düşük dereceye doğru sıraladık.
    root_s = np.roots(a_l)
    print(f'root_s: {root_s.shape}')
    #for item in range(D):
    for element in range(len(l)-1):
        genlik[element] = abs(root_s[element])#köklerin genliklerini ayrı bir dizide tutuyoruz.
    
    final = np.array([0 for i in range(D*2)],dtype=np.complex_) #birim çembere yakın kökleri tutacağımız final dizisi.
    #bu final dizisinin veri tipini complex seçtik çünkü rootlar complex.
    
    #Birim çembere yakın kökleri bulmak için genliği 1 e en yakın olanları seçeceğiz. Aşağıdaki döngüyle bunu yapmaktayız.
    #M tane sinyal için 4 tane kök seçmemizin sebebi her açıdan 2 adet bulmasından kaynaklanmaktadır. Bu yüzden sonuçlar örneğin
    #2 açı için 4 tane olacak şekilde çıkacaktır. Döngüde önce 1 e en yakın olan değer bulunur sonra bunun root_s arrayindeki
    #indeksi bulunur. Daha sonra bu indeksteki kök final arrayine atılır ve genlik arrayindeki aynı indeksli eleman silinir.
    #Bu silme işleminin sebebi döngünün bir sonraki kısmında 1 e en yakın 2. elemanı bulabilmesi içindir. Bu şekilde her adımda
    #genlik arrayinden o adımdaki eleman silinir. Ancak root_s deki veriler silinmez. Bu döngü böyle çalışmaktadır.
    for item in range(D*2):
        nearest = closest(genlik,1)
        index_of_angle = np.where(abs(root_s) == nearest)
        final[item] = root_s[index_of_angle[0][0]]
        index_of_angle = np.where(genlik == nearest)
        genlik = np.delete(genlik, index_of_angle)
    
    #Alttaki döngü ise ROOT-MUSIC algoritmasındaki sin^-1 li formülün uygulamasını yapar. Elde edilen kompleks köklerin
    #açıya döndürülmesini sağlar. En son final arrayini float olan ve açıları tutan bir array haline getiririz.
    for item in range(len(final)):
        faz = cmath.phase(final[item])
        final[item] = np.rad2deg(np.arcsin(lamda*faz/(2*np.pi*delta)))
    final = np.array(final,dtype=np.float_)
    
    return final


# # Güç hesabı böyle yapılabilir
# t = np.linspace(1, 100, 1000)
# x_volts = 5*np.sin(t/(2*np.pi))
# x_watts = x_volts ** 2
# #print(x_watts)
# sum(x_watts)/len(x_watts)

# In[ ]:





# In[ ]:




