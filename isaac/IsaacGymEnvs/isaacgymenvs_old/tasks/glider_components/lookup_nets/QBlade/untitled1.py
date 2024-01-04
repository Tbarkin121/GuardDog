

# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 20:17:35 2023

@author: MiloPC
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import time

plt.style.use('dark_background')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.set_default_device('cuda')

#%%
AlphaDeg = 5
BetaDeg = 0
plot_scale = 3.
m = 100000
plot_id = 0

#%%


naca_m =    [0.00,
             0.00]

naca_p =    [0.0,
             0.0]

nc =        [10]

y_mirror =  [True]

translate = [torch.tensor([0., 0., 0.])]

sections = [torch.tensor([[00.0,  00.0,   00.0,   5.0],
                          [0.0,  10.0,   5.0,   5.0]])]

ns =        [torch.tensor([10])]

ctrl_id = [[[0.25,torch.tensor([.1,.2]),0.,torch.tensor([0,1,0])],
            [0.25,torch.tensor([.3,.4]),0.,torch.tensor([0,1,0])]]]

#%%

one_over_4pi = 1/4/torch.pi
alpha = torch.tensor(AlphaDeg*torch.pi/180.)
beta = torch.tensor(BetaDeg*torch.pi/180.)
cosA =torch.cos(alpha)
sinA =torch.sin(alpha)
cosB =torch.cos(beta)
sinB =torch.sin(beta)

TX = torch.tensor([[1.,0.,0.],[0.,cosB,-sinB],[0.,sinB,cosB]])
TY = torch.tensor([[cosA,0.,sinA],[0.,1.,0.],[-sinA,0.,cosA]])

    

T_s = torch.matmul(TY,TX)
r = torch.zeros([1,0,1,3])
r_a = torch.zeros([1,1,0,3])
r_b = torch.zeros([1,1,0,3])

sref = 0
gxj = []
gyj = []
n_hat = torch.zeros([0,3])

grid_x = []
grid_y = []
grid_z = []
cntrd_x_np = []
cntrd_y_np = []
cntrd_z_np = []
cntrd_x_np_n1 = []
cntrd_y_np_n1 = []
cntrd_z_np_n1 = []
n_hat_j_np = []
n_hat_j_plo_np = []

n = 0
strips = 0

for j in range(len(nc)):
    
    
    if y_mirror[j]:
        mirrored_section = sections[j][1:,:].flipud()
        mirrored_section[:,1] = -sections[j][1:,1]
        sections[j] = torch.cat([mirrored_section,sections[j]],dim=0)
        ns[j] = torch.cat([ns[j].flip(dims=[0]),ns[j]],dim=0)
        
        for k in range(len(ctrl_id[j])):
            ctrl_id_mirrored = ctrl_id[j][k][:]
            ctrl_id_mirrored[1] = 1 - ctrl_id_mirrored[1]
            ctrl_id[j] += [ctrl_id_mirrored]
        
    
    strips += torch.sum(ns[j])
    n+=torch.sum(nc[j]*ns[j])
    
    xj_ofs = sections[j][:,0]
    yj_ofs = sections[j][:,1]
    zj_ofs = sections[j][:,2]
    cj = sections[j][:,3]
    
    syj = yj_ofs.diff().abs()
    szj = zj_ofs.diff().abs()
    

    sxz_j = (syj**2 + szj**2).sqrt()
    
    
    sref += torch.sum(0.5*(cj[:-1] + cj[1:])*sxz_j)
    
    naca_x0 = torch.linspace(0,1,nc[j]+1)+(1/(nc[j]+1)/4)
    naca_x = 0.5*(naca_x0[0:-1]+naca_x0[1:])
    
    naca_term = 2*naca_m[j]*(naca_p[j] - naca_x)
    dy_dx = naca_term/naca_p[j]**2
    dyb_dx = naca_term/(naca_p[j]-1.)**2
    
    dy_dx[naca_x>naca_p[j]] = dyb_dx[naca_x>naca_p[j]]
    
    n_vec = torch.concat([-dy_dx.reshape([-1,1]),torch.zeros([nc[j],1]),torch.ones([nc[j],1])],dim=1)
    n_hat_j = n_vec/torch.sqrt(torch.sum(n_vec**2,dim=1)).reshape([-1,1])
    n_hat_j_plo = torch.zeros([nc[j],3])
    n_hat_j_plo[:,2]= 1.
    
    gxn,gyn = torch.meshgrid(torch.linspace(0,1,nc[j]+1),torch.linspace(0,1,ns[j][0]+1))
    gxj = gxn*torch.linspace(cj[0],cj[1],ns[j][0]+1).reshape(1,ns[j][0]+1) + torch.linspace(xj_ofs[0],xj_ofs[1],ns[j][0]+1).reshape(1,ns[j][0]+1)
    gyj = gyn*syj[0] 
    gzj = gyj*0 + torch.linspace(zj_ofs[0],zj_ofs[1],ns[j][0]+1).reshape(1,ns[j][0]+1)
    
    si = 0
    
    ang = torch.atan2(zj_ofs[1] - zj_ofs[0],syj[0])
    
    cosA =torch.cos(ang)
    sinA =torch.sin(ang)
    T_sx = torch.tensor([[1.,0.,0.],[0.,cosA,-sinA],[0.,sinA,cosA]])
    
    n_hat_ji = torch.matmul(T_sx, n_hat_j.permute([1,0])).reshape(3,-1,1)*torch.ones([1,1,ns[j][0]])
    n_hat_ji_plo = torch.matmul(T_sx, n_hat_j_plo.permute([1,0])).reshape(3,-1,1)*torch.ones([1,1,ns[j][0]])
    
    for i in range(syj.shape[0]-1):
        
        ang = torch.atan2(zj_ofs[i+2] - zj_ofs[i+1],syj[i+1])
        
        cosA =torch.cos(ang)
        sinA =torch.sin(ang)
        T_sx = torch.tensor([[1.,0.,0.],[0.,cosA,-sinA],[0.,sinA,cosA]])
        
        gxn,gyn = torch.meshgrid(torch.linspace(0,1,nc[j]+1),torch.linspace(0,1,ns[j][i+1]+1))
        gxi = gxn*torch.linspace(cj[i+1],cj[i+2],ns[j][i+1]+1).reshape(1,ns[j][i+1]+1) + torch.linspace(xj_ofs[i+1],xj_ofs[i+2],ns[j][i+1]+1).reshape(1,ns[j][i+1]+1)
        si += syj[i] 
        gyi = gyn*syj[i+1] + si
        gzi = gyi*0 + torch.linspace(zj_ofs[i+1],zj_ofs[i+2],ns[j][i+1]+1).reshape(1,ns[j][i+1]+1)
    
        gxj = torch.concat([gxj,gxi[:,1:]],dim=1)
        gyj = torch.concat([gyj,gyi[:,1:]],dim=1)
        gzj = torch.concat([gzj,gzi[:,1:]],dim=1)

        n_hat_ji_plo = torch.concat([n_hat_ji_plo,torch.matmul(T_sx, n_hat_j_plo.permute([1,0])).reshape(3,-1,1)*torch.ones([1,1,ns[j][i+1]])],dim=2)
        n_hat_ji = torch.concat([n_hat_ji,torch.matmul(T_sx, n_hat_j.permute([1,0])).reshape(3,-1,1)*torch.ones([1,1,ns[j][i+1]])],dim=2)
        
    n_j = (gxj.shape[1]-1)*(nc[j])
    
    gxj += translate[j][0] 
    gyj += translate[j][1] 
    gzj += translate[j][2] 
    
    n_hat_ji_plo = n_hat_ji_plo.permute([1,2,0])
    n_hat_ji = n_hat_ji.permute([1,2,0])
    
    gja = torch.cat([gxj[0:-1,0:-1].unsqueeze(0),gyj[0:-1,0:-1].unsqueeze(0),gzj[0:-1,0:-1].unsqueeze(0)],dim=0).reshape(3,n_j).permute([1,0]).unsqueeze(2)
    gjb = torch.cat([gxj[1:,0:-1].unsqueeze(0),gyj[1:,0:-1].unsqueeze(0),gzj[1:,0:-1].unsqueeze(0)],dim=0).reshape(3,n_j).permute([1,0]).unsqueeze(2)
    gjc = torch.cat([gxj[0:-1,1:].unsqueeze(0),gyj[0:-1,1:].unsqueeze(0),gzj[0:-1,1:].unsqueeze(0)],dim=0).reshape(3,n_j).permute([1,0]).unsqueeze(2)
    gjd = torch.cat([gxj[1:,1:].unsqueeze(0),gyj[1:,1:].unsqueeze(0),gzj[1:,1:].unsqueeze(0)],dim=0).reshape(3,n_j).permute([1,0]).unsqueeze(2)
   
    A = torch.cat([gjd-gja,gjc-gjb],dim=2)
    B = gjb-gja
    
    t = torch.linalg.lstsq(A,B)[0].permute([0,2,1])

    X = ((gjd-gja).squeeze()*t[:,:,0] + gja.squeeze()).permute([1,0]).reshape(3,nc[j],ns[j].sum())
    
    cntrd_x = X[0]
    cntrd_y = X[1]
    cntrd_z = X[2]

    grid_x += [gxj]
    grid_y += [gyj]
    grid_z += [gzj]
    cntrd_x_np += [cntrd_x.cpu().numpy()]
    cntrd_y_np += [cntrd_y.cpu().numpy()]
    cntrd_z_np += [cntrd_z.cpu().numpy()]
    cntrd_x_np_n1 += [cntrd_x.reshape([-1,1]).cpu().numpy()]
    cntrd_y_np_n1 += [cntrd_y.reshape([-1,1]).cpu().numpy()]
    cntrd_z_np_n1 += [cntrd_z.reshape([-1,1]).cpu().numpy()]
    
    n_hat_j_np += [n_hat_ji.cpu().numpy()]
    n_hat_j_plo_np += [n_hat_ji_plo.cpu().numpy()]
    
    r_j = torch.concat([cntrd_x.reshape([-1,1]),cntrd_y.reshape([-1,1]),cntrd_z.reshape([-1,1])],dim = 1).reshape(1,n_j,1,3)

    r_a_j = torch.concat([gxj[0:-1,0:-1].reshape([-1,1]),gyj[0:-1,0:-1].reshape([-1,1]),gzj[0:-1,0:-1].reshape([-1,1])],dim = 1).reshape(1,1,n_j,3)
    r_b_j = torch.concat([gxj[0:-1,1:].reshape([-1,1]),gyj[0:-1,1:].reshape([-1,1]),gzj[0:-1,1:].reshape([-1,1])],dim = 1).reshape(1,1,n_j,3)
    
    r = torch.concat([r,r_j],dim=1)
    r_a = torch.concat([r_a,r_a_j],dim=2)
    r_b = torch.concat([r_b,r_b_j],dim=2)
    
    n_hat = torch.concat([n_hat,n_hat_ji.reshape([-1,3])],dim=0)
    
    
sref = 100.0
    
grid_xyz = [grid_x,grid_y,grid_z]    
cntrd_xyz = [cntrd_x_np,cntrd_y_np,cntrd_z_np]    
cntrd_xyz_n1 = [cntrd_x_np_n1,cntrd_y_np_n1,cntrd_z_np_n1]     
n_hat = n_hat.unsqueeze(0).unsqueeze(0)
#%%

U_vel = torch.zeros([m,1,n,3])
U_vel[:,:,:,0] = -torch.cos(alpha)*torch.cos(beta)
U_vel[:,:,:,1] = torch.sin(beta)
U_vel[:,:,:,2] = -torch.sin(alpha)*torch.cos(beta)

x_hat = torch.zeros(1,1,n,3)
x_hat[0,0,:,0] = 1 

a = r - r_a
b = r - r_b

norm_a = torch.reshape(torch.sqrt(torch.sum(a**2,dim=3)),[1,n,n,1])
norm_b = torch.reshape(torch.sqrt(torch.sum(b**2,dim=3)),[1,n,n,1])

a_cross_b = torch.cross(a,b)
a_dot_b = torch.reshape(torch.sum(a*b,dim=3),[1,n,n,1])

a_cross_x_hat = torch.cross(a,x_hat)
b_cross_x_hat = torch.cross(b,x_hat)

a_dot_x_hat = torch.reshape(torch.sum(a*x_hat,dim=3),[1,n,n,1])
b_dot_x_hat = torch.reshape(torch.sum(b*x_hat,dim=3),[1,n,n,1])

Vi = one_over_4pi*(a_cross_b/(norm_a*norm_b + a_dot_b)*(1/norm_a + 1/norm_b) + a_cross_x_hat/(norm_a - a_dot_x_hat)*(1/norm_a) - b_cross_x_hat/(norm_b - b_dot_x_hat)*(1/norm_b))

ri = 0.5*(r_a + r_b).permute([0,2,1,3])

a = ri - r_a
b = ri - r_b

norm_a = torch.reshape(torch.sqrt(torch.sum(a**2,dim=3)),[1,n,n,1])
norm_b = torch.reshape(torch.sqrt(torch.sum(b**2,dim=3)),[1,n,n,1])

a_cross_b = torch.cross(a,b)
a_dot_b = torch.reshape(torch.sum(a*b,dim=3),[1,n,n,1])

a_cross_x_hat = torch.cross(a,x_hat)
b_cross_x_hat = torch.cross(b,x_hat)

a_dot_x_hat = torch.reshape(torch.sum(a*x_hat,dim=3),[1,n,n,1])
b_dot_x_hat = torch.reshape(torch.sum(b*x_hat,dim=3),[1,n,n,1])

# Vi_ri = one_over_4pi*(a_cross_b/(norm_a*norm_b + a_dot_b)*(1/norm_a + 1/norm_b) + a_cross_x_hat/(norm_a - a_dot_x_hat)*(1/norm_a) - b_cross_x_hat/(norm_b - b_dot_x_hat)*(1/norm_b))
Vi_ri = one_over_4pi*(a_cross_x_hat/(norm_a - a_dot_x_hat)*(1/norm_a) - b_cross_x_hat/(norm_b - b_dot_x_hat)*(1/norm_b))

Vi_ri[torch.isnan(Vi_ri)] = 0.



Vi_dot_n_hat = torch.sum(Vi*n_hat.reshape([1,-1,1,3]),dim = 3)


Vi = Vi.permute(3,1,2,0).squeeze(3)
Vi_ri = Vi_ri.permute(3,1,2,0).squeeze(3)

Vi_infl = Vi_dot_n_hat.reshape(n,n)

li = (r_b - r_a)

LU, pivots = torch.linalg.lu_factor(Vi_infl)

#%%
n_component_old = 0

delta_ctrl = torch.zeros([n,0])

for component in range(len(ctrl_id)):
    
    n_strip_comp = torch.sum(ns[component])
    
    idx_mat = torch.linspace(0,(n_strip_comp)*(nc[component])-1,(n_strip_comp)*(nc[component])).reshape([nc[component],n_strip_comp])
    for surface in range(len(ctrl_id[component])):

        naca_x0 = torch.linspace(0,1,nc[component]+1)+(1/(nc[component]+1)/4)
        naca_x = 0.5*(naca_x0[0:-1]+naca_x0[1:])
        
        if nc[component]>1:
            xid = torch.min((naca_x > (1-ctrl_id[component][surface][0])).nonzero())
        else:
            xid = 0
        
        ysn = grid_xyz[1][component][0]/grid_xyz[1][component][0,-1]
        ysn_cntrd = 0.5*(ysn[:-1]+ysn[1:])
        
        y_prcnt_a = ctrl_id[component][surface][1][0]
        y_prcnt_b = ctrl_id[component][surface][1][1]
        
        yid = (((ysn_cntrd > y_prcnt_a)*1.0 + (ysn_cntrd < y_prcnt_b))-1).nonzero()
        idx_surf = idx_mat[xid:,yid].reshape([-1,1]) + n_component_old
        
        ctrl_hot = torch.zeros([n,1])
        ctrl_hot[idx_surf.int(),0] = 1

        delta_ctrl = torch.concat([delta_ctrl,ctrl_hot],dim=1)
        
    n_component_old += (n_strip_comp)*(nc[component])
    
    
#%%
ctrl_vec = torch.zeros([1,delta_ctrl.shape[1]])
counter = 0
for component in range(len(ctrl_id)):
    for surface in range(len(ctrl_id[component])):
        ctrl_vec[0,surface] = torch.tensor(ctrl_id[component][surface][2])*torch.pi/180.

ctrl_mat = ctrl_vec*torch.ones([m,1,1])

# ctrl_mat = 0.1*(torch.rand([m,1,6])*2.0 - 1.0)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

now = time.perf_counter()

ctrl_tns = delta_ctrl*ctrl_mat
ctrl_RHS = ctrl_tns.sum(dim=2).permute([1, 0])
RHS = torch.sum(U_vel*n_hat,dim = 3).permute([2,0,1]).squeeze() + ctrl_RHS
Gamma = torch.linalg.lu_solve(LU, pivots, RHS)
V = torch.matmul(Vi_ri,Gamma.unsqueeze(0)).permute(2,1,0).unsqueeze(1) - U_vel
Fi = 2*torch.cross(V,li)*Gamma.permute([1,0]).unsqueeze(1).unsqueeze(3)/sref
F = (Fi).sum(dim=2).permute([2,0,1]).squeeze()
C_DYL = torch.matmul(T_s, F)

dt = time.perf_counter() - now

CDi = -torch.sum(U_vel[0,0,0,:]*F[:,0])

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

C_DYL_np = C_DYL.cpu().numpy()

print('-------------------------------------------------------------------')
print('# Surfaces =',len(nc))
print('# Strips   =',strips.cpu().numpy())
print('# Vortices =',n.cpu().numpy())
print('')
print('Sref       =',sref)
print('')
print('Alpha      =',AlphaDeg)
print('Beta       =',BetaDeg)
print('')
print('CXtot      =',F[0,0].cpu().numpy())
print('CYtot      =',F[1,0].cpu().numpy())
print('CZtot      =',F[2,0].cpu().numpy())
print('')
print('CLtot      =',C_DYL_np[2,0])
print('CDind      =',C_DYL_np[0,0])
print('')
print('SPS        =',m/dt)
print('------------------------------------------------------------------')
plt.close('all')
fig = plt.figure(figsize=plt.figaspect(1))
ax = fig.add_subplot(1, 1, 1, projection='3d')
# ax = fig.add_subplot(2, 1, 1, projection='3d')

for i in range(len(ns)):
    ax.plot_wireframe(grid_xyz[0][i].cpu().numpy(),
                      grid_xyz[1][i].cpu().numpy(),
                      grid_xyz[2][i].cpu().numpy(),color='white', linewidth = 0.5)
    ax.quiver(cntrd_xyz_n1[0][i],
              cntrd_xyz_n1[1][i],
              cntrd_xyz_n1[2][i],
              n_hat_j_np[i][:,:,0].reshape([-1,1]),
              n_hat_j_np[i][:,:,1].reshape([-1,1]),
              n_hat_j_np[i][:,:,2].reshape([-1,1]),
              length=0.1, normalize=True, color = 'blue', linewidth = 1)

for component in range(len(ctrl_id)):
    for surface in range(len(ctrl_id[component])):
        naca_x0 = torch.linspace(0,1,nc[component]+1)+(1/(nc[component]+1)/4)
        naca_x = 0.5*(naca_x0[0:-1]+naca_x0[1:])
        if nc[component]>1:
            xid = torch.min((naca_x > (1-ctrl_id[component][surface][0])).nonzero())
        else:
            xid = 0
        ysn = grid_xyz[1][component][0]/grid_xyz[1][component][0,-1]
        
        ysn_cntrd = 0.5*(ysn[:-1]+ysn[1:])
        
        y_prcnt_a = ctrl_id[component][surface][1][0]
        y_prcnt_b = ctrl_id[component][surface][1][1]
        
        yid = (((ysn_cntrd > y_prcnt_a)*1.0 + (ysn_cntrd < y_prcnt_b))-1).nonzero()

        yid_a = torch.min(yid)
        yid_b = torch.max(yid)+1
     
        ax.plot_surface(grid_xyz[0][component][xid:,yid_a:yid_b+1].cpu().numpy(),
                        grid_xyz[1][component][xid:,yid_a:yid_b+1].cpu().numpy(),
                        grid_xyz[2][component][xid:,yid_a:yid_b+1].cpu().numpy(),color=[1.,0.,1.], linewidth = 1)

ax.set_aspect('equal')
ax.set_proj_type('ortho')
ax.view_init(elev=20, azim=225)
# Transparent spines
ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

# Transparent panes
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

# No ticks
ax.set_xticks([]) 
ax.set_yticks([]) 
ax.set_zticks([])

#%%

fig = plt.figure(figsize=plt.figaspect(1))
ax = fig.add_subplot(1, 1, 1, projection='3d')
# ax = fig.add_subplot(2, 1, 2, projection='3d')

idx_a = 0

for component in range(len(ns)):

    idx_b = idx_a + (torch.sum(ns[component]))*(nc[component])
    
    Gamma0_plo = (Gamma[idx_a:idx_b,plot_id].reshape([nc[component],-1])*plot_scale).cpu().numpy()

    idx_a = idx_b+0
    ax.plot_wireframe(grid_xyz[0][component].cpu().numpy(),
                      grid_xyz[1][component].cpu().numpy(),
                      grid_xyz[2][component].cpu().numpy(),color=[1.,0.,1.],linewidth=0.5)
    
    for strip in range((grid_xyz[0][component].shape[1]-1)):
        ax.plot_wireframe(cntrd_xyz[0][component][:,strip]*np.ones([2,1]) + Gamma0_plo[:,strip]*np.array([[0.],[1.]])*n_hat_j_plo_np[component][:,strip,0],
                          cntrd_xyz[1][component][:,strip]*np.ones([2,1]) + Gamma0_plo[:,strip]*np.array([[0.],[1.]])*n_hat_j_plo_np[component][:,strip,1],
                          cntrd_xyz[2][component][:,strip]*np.ones([2,1]) + Gamma0_plo[:,strip]*np.array([[0.],[1.]])*n_hat_j_plo_np[component][:,strip,2],
                          color=[0.,1.,0.],linewidth=0.5)
        
        ax.plot_wireframe(cntrd_xyz[0][component][:,strip]*np.ones([2,1]) + Gamma0_plo[:,strip]*np.array([[1.],[1.]])*n_hat_j_plo_np[component][:,strip,0],
                          cntrd_xyz[1][component][:,strip]*np.ones([2,1]) + Gamma0_plo[:,strip]*np.array([[1.],[1.]])*n_hat_j_plo_np[component][:,strip,1],
                          cntrd_xyz[2][component][:,strip]*np.ones([2,1]) + Gamma0_plo[:,strip]*np.array([[1.],[1.]])*n_hat_j_plo_np[component][:,strip,2],
                          color=[1.,0.,0.],linewidth=0.5)
        
        


for component in range(len(ctrl_id)):
    for surface in range(len(ctrl_id[component])):
        naca_x0 = torch.linspace(0,1,nc[component]+1)+(1/(nc[component]+1)/4)
        naca_x = 0.5*(naca_x0[0:-1]+naca_x0[1:])
        
        if nc[component]>1:
            xid = torch.min((naca_x > (1-ctrl_id[component][surface][0])).nonzero())
        else:
            xid = 0
        ysn = grid_xyz[1][component][0]/grid_xyz[1][component][0,-1]
        
        ysn_cntrd = 0.5*(ysn[:-1]+ysn[1:])
        
        y_prcnt_a = ctrl_id[component][surface][1][0]
        y_prcnt_b = ctrl_id[component][surface][1][1]
        
        yid = (((ysn_cntrd > y_prcnt_a)*1.0 + (ysn_cntrd < y_prcnt_b))-1).nonzero()

        yid_a = torch.min(yid)
        yid_b = torch.max(yid)+1
        
        ax.plot_wireframe(grid_xyz[0][component][xid:,yid_a:yid_b+1].cpu().numpy(),
                          grid_xyz[1][component][xid:,yid_a:yid_b+1].cpu().numpy(),
                          grid_xyz[2][component][xid:,yid_a:yid_b+1].cpu().numpy(),color=[1.,1.,1.], linewidth = 1)

ax.set_aspect('equal')
ax.set_proj_type('ortho')
ax.view_init(elev=20, azim=225)
# Transparent spines
ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

# Transparent panes
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

# No ticks
ax.set_xticks([]) 
ax.set_yticks([]) 
ax.set_zticks([])

#%%

# start_x = -5
# gamma0 = Gamma[:,0].reshape(1,1,-1,1)
# step_const = 0.1
# n_stream_y = 6
# n_stream_z = 6

# ry,rz = torch.meshgrid(4.*torch.linspace(-1.0,1.0,n_stream_y),4.*torch.linspace(-1.,1.,n_stream_z))

# start_xyz = torch.concat([start_x*torch.ones([n_stream_z*n_stream_y,1]),ry.reshape([-1,1]),rz.reshape([-1,1])],dim = 1).reshape(1,-1,1,3)

# nt = 100
# streams = torch.zeros(nt,1,n_stream_z*n_stream_y,1,3)
# streams[0,:] = start_xyz;

# U_vel_n = torch.zeros([1,n_stream_z*n_stream_y,3])
# U_vel_n[:,:,0] = torch.cos(alpha)*torch.cos(Beta)
# U_vel_n[:,:,1] = torch.sin(Beta)
# U_vel_n[:,:,2] = torch.sin(alpha)*torch.cos(Beta)

# for i in range(nt-1):
    
    
#     r_b = li+r_a
    
#     a = streams[i,:] - r_a
#     b = streams[i,:] - r_b
    
#     norm_a = torch.reshape(torch.sqrt(torch.sum(a**2,dim=3)),[1,n_stream_z*n_stream_y,n,1])
#     norm_b = torch.reshape(torch.sqrt(torch.sum(b**2,dim=3)),[1,n_stream_z*n_stream_y,n,1])
    
#     a_cross_b = torch.cross(a,b)
#     a_dot_b = torch.reshape(torch.sum(a*b,dim=3),[1,n_stream_z*n_stream_y,n,1])
    
#     a_cross_x_hat = torch.cross(a,x_hat)
#     b_cross_x_hat = torch.cross(b,x_hat)
    
#     a_dot_x_hat = torch.reshape(torch.sum(a*x_hat,dim=3),[1,n_stream_z*n_stream_y,n,1])
#     b_dot_x_hat = torch.reshape(torch.sum(b*x_hat,dim=3),[1,n_stream_z*n_stream_y,n,1])
    
#     Vi = one_over_4pi*(a_cross_b/(norm_a*norm_b + a_dot_b)*(1/norm_a + 1/norm_b) + 
#                         a_cross_x_hat/(norm_a - a_dot_x_hat)*(1/norm_a) - 
#                         b_cross_x_hat/(norm_b - b_dot_x_hat)*(1/norm_b))
#     V = torch.sum(Vi*gamma0,dim=2) + U_vel_n
    
#     streams[i+1,:] = streams[i,:] + V.reshape(1,-1,1,3)*step_const

# streams_cpu = streams.cpu().numpy()

# i = 0

# for i in range(n_stream_z*n_stream_y):
    
#     stream_xi = streams_cpu[:,0,i,0,0]
#     stream_yi = streams_cpu[:,0,i,0,1]
#     stream_zi = streams_cpu[:,0,i,0,2]
    
#     ax.plot3D(stream_xi,stream_yi,stream_zi,'c')
    
# ax.set_aspect('equal')