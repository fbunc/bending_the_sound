plt.rcParams.update({
        "lines.color": "black",
        "patch.edgecolor": "black",
        "text.color": "black",
        "axes.facecolor": "black",
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "grid.color": "black",
        "figure.facecolor": "black",
        "figure.edgecolor": "black",
        "savefig.facecolor": "black",
        "savefig.edgecolor": "black"})


fig = plt.figure(figsize=(20,20))
hsvwheel = cm.get_cmap('hsv', To)
ax = fig.add_subplot(111, projection='3d')
#fig.set_size_inches(20, 20)
ax.grid(False)
ax.w_xaxis.pane.fill = False
ax.w_yaxis.pane.fill = False
ax.w_zaxis.pane.fill = False
plt.style.use('dark_background')
path_out='/content/drive/MyDrive/'
text='ttttest'
fps=2

ims = []

# Each index element has 8 internal steps able to model different dynamics
# The size of the symbol can indicate the first/second/third different
# The color is a discrete mapping in HSV 
z_carrier_orig=z_carrier
k=-1
#z_carrier=z_carrier**(-k)
 #prime symbol size
b_size=39*13
T_color=39*2
T_mod=T_color
hsvwheel = cm.get_cmap('hsv', T_color)

nmod = events_index%T_mod
symbol=nmod
for n in np.arange(z_carrier[0:b_size].size):
  
  if isPrime(n):
    event_marker="o"
    size=(13)**(0.5*(1+np.sqrt(5)))
    ax.scatter(z_carrier_alpha[n],z_carrier_beta[n],np.log(np.abs(z_carrier[n])),color=hsvwheel(symbol[n-7]),s=size,marker=event_marker)
    plt.savefig('/content/drive/MyDrive/out_primes_entangled/np_A_%d.png' % n)
    ax.scatter((+1)*z_carrier_alpha[n],(-1)*z_carrier_beta[n],np.log(np.abs(z_carrier[n])),color=hsvwheel(symbol[n-6]),s=size,marker=event_marker)
    plt.savefig('/content/drive/MyDrive/out_primes_entangled/np_B_%d.png' % n)
    ax.scatter((-1)*z_carrier_alpha[n],(+1)*z_carrier_beta[n],np.log(np.abs(z_carrier[n])),color=hsvwheel(symbol[n-5]),s=size,marker=event_marker)
    plt.savefig('/content/drive/MyDrive/out_primes_entangled/np_C_%d.png' % n)
    ax.scatter((-1)*z_carrier_alpha[n],(-1)*z_carrier_beta[n],np.log(np.abs(z_carrier[n])),color=hsvwheel(symbol[n-4]),s=size,marker=event_marker)
    plt.savefig('/content/drive/MyDrive/out_primes_entangled/np_D_%d.png' % n)
    ax.scatter(z_carrier_beta[n],z_carrier_alpha[n],np.log(np.abs(z_carrier[n])),color=hsvwheel(symbol[n-3]),s=size,marker=event_marker)
    plt.savefig('/content/drive/MyDrive/out_primes_entangled/np_E_%d.png' % n)
    ax.scatter((+1)*z_carrier_beta[n],(-1)*z_carrier_alpha[n],np.log(np.abs(z_carrier[n])),color=hsvwheel(symbol[n-2]),s=size,marker=event_marker)
    plt.savefig('/content/drive/MyDrive/out_primes_entangled/np_F_%d.png' % n)
    ax.scatter((-1)*z_carrier_beta[n],(+1)*z_carrier_alpha[n],np.log(np.abs(z_carrier[n])),color=hsvwheel(symbol[n-1]),s=size,marker=event_marker)
    plt.savefig('/content/drive/MyDrive/out_primes_entangled/np_G_%d.png' % n)
    ax.scatter((-1)*z_carrier_beta[n],(-1)*z_carrier_alpha[n],np.log(np.abs(z_carrier[n])),color=hsvwheel(symbol[n-0]),s=1.5*size,marker=event_marker)
    plt.savefig('/content/drive/MyDrive/out_primes_entangled/p_H_%d.png' % n)
  else:
    event_marker="o"
    size=13*2.17
    ax.scatter(z_carrier_alpha[n],z_carrier_beta[n],np.log(np.abs(z_carrier[n])),color=hsvwheel(symbol[n-7]),s=1.5*size,marker=event_marker)
    plt.savefig('/content/drive/MyDrive/out_primes_entangled/np_A_%d.png' % n)
    ax.scatter((+1)*z_carrier_alpha[n],(-1)*z_carrier_beta[n],np.log(np.abs(z_carrier[n])),color=hsvwheel(symbol[n-6]),s=size,marker=event_marker)
    plt.savefig('/content/drive/MyDrive/out_primes_entangled/np_B_%d.png' % n)
    ax.scatter((-1)*z_carrier_alpha[n],(+1)*z_carrier_beta[n],np.log(np.abs(z_carrier[n])),color=hsvwheel(symbol[n-5]),s=size,marker=event_marker)
    plt.savefig('/content/drive/MyDrive/out_primes_entangled/np_C_%d.png' % n)
    ax.scatter((-1)*z_carrier_alpha[n],(-1)*z_carrier_beta[n],np.log(np.abs(z_carrier[n])),color=hsvwheel(symbol[n-4]),s=size,marker=event_marker)
    plt.savefig('/content/drive/MyDrive/out_primes_entangled/np_D_%d.png' % n)
    ax.scatter(z_carrier_beta[n],z_carrier_alpha[n],np.log(np.abs(z_carrier[n])),color=hsvwheel(symbol[n-3]),s=size,marker=event_marker)
    plt.savefig('/content/drive/MyDrive/out_primes_entangled/np_E_%d.png' % n)
    ax.scatter((+1)*z_carrier_beta[n],(-1)*z_carrier_alpha[n],np.log(np.abs(z_carrier[n])),color=hsvwheel(symbol[n-2]),s=size,marker=event_marker)
    plt.savefig('/content/drive/MyDrive/out_primes_entangled/np_F_%d.png' % n)
    ax.scatter((-1)*z_carrier_beta[n],(+1)*z_carrier_alpha[n],np.log(np.abs(z_carrier[n])),color=hsvwheel(symbol[n-1]),s=size,marker=event_marker)
    plt.savefig('/content/drive/MyDrive/out_primes_entangled/np_G_%d.png' % n)
    ax.scatter((-1)*z_carrier_beta[n],(-1)*z_carrier_alpha[n],np.log(np.abs(z_carrier[n])),color=hsvwheel(symbol[n-0]),s=size,marker=event_marker)
    plt.savefig('/content/drive/MyDrive/out_primes_entangled/np_H_%d.png' % n)

   
  #append current plot to ims
  #fig.canvas.draw()
  #im = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
  #ims.append([im.reshape(-1,1)])
  plt.savefig('/content/drive/MyDrive/out_primes_entangled/eight_entangled_events_time_slice_%d.png' % n)

#ani = animation.ArtistAnimation(fig, ims)
#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=fps)
#ani.save(f'{path_out}{text}.mp4', writer=writer)
#ani.save(f'{path_out}{text}.mp4', writer=writer)
plt.show()



  
  #plt.savefig(f'/content/drive/MyDrive/eigen_hole_{n}.png',dpi=dots_per_inch)
  

