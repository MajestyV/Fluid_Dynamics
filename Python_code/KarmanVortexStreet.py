import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

class LBM_KVS:
    ''' This code is designed for simulating the Karman vortex street using the lattice Boltzmann method. '''
    def __init__(self,**kwargs):
        # 本代码采用无量纲的流体动力学框架，在此框架下，dx = dy = dt = 1.0（一个lattice units）
        # 设置管道形状
        lx = kwargs['length'] if 'length' in kwargs else 801  # 管道的长
        ly = kwargs['width'] if 'width' in kwargs else  201   # 管道的宽
        self.lx, self.ly = [lx, ly]                           # 转换成实例变量
        # 设置管道边界条件：[left,top,right,bottom] boundary conditions: 0 -> Dirichlet ; 1 -> Neumann
        boundary_type = kwargs['boundary_condition'] if 'boundary_condition' in kwargs else [0, 0, 1, 0]
        # 设置边界速度，此代码模拟的是二维的管道（区域）中的流体，所以速度具有两个自由度，即流体速度可分解为沿x方向的Vx和沿y方向的Vy
        boundary_vel = kwargs['boundary_initial'] if 'boundary_initial' in kwargs else [[0.1, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        self.bc_type = np.array(boundary_type)      # 转换为数组，方便后续计算
        self.bc_vel = np.array(boundary_vel)        # 转换为二维数组（亦可理解为二阶张量），shape应为(4,2)
        # 设置障碍物位置大小参数
        self.cylinder = kwargs['cylinder'] if 'cylinder' in kwargs else 'True'  # 选择是否要放置圆柱体障碍物
        cylinder_param = kwargs['cylinder_parameter'] if 'cylinder_parameter' in kwargs else [160.0, 100.0, 20.0]  # 圆柱体参数：(x轴坐标, y轴坐标, 半径)
        self.cy_param = np.array(cylinder_param)    # 转换为数组，方便计算
        self.obstacle = np.empty((lx, ly))

        # 设置无量纲流体动力学参数
        self.nu = kwargs['viscosity'] if 'viscosity' in kwargs else 0.01  # viscosity of fluid (粘滞系数)
        self.tau = 3.0*self.nu+0.5                                        # 弛豫时间τ
        self.inv_tau = 1.0/self.tau                                       # 先算好弛豫时间τ的倒数，减少计算量

        # 建立二维九速四方格子的格点玻尔兹曼模型（D2Q9 LBM）
        self.coord = [(x,y) for x in range(lx) for y in range(ly)]                 # 管道模型的所有坐标
        self.coord_inner = [(x,y) for x in range(1,lx-1) for y in range(1,ly-1)]   # 管道内部空间（除去边界）的所有坐标
        self.rho = np.empty((lx,ly))      # 创建一个lx行，ly列的二阶张量以存放每个格点上的流体密度
        self.vel = np.empty((lx,ly,2))    # 创建一个形状为 (lx,ly,2) 的三阶张量来存放每个格点上的流体速度（对于二维系统，流体速度有两个自由度）
        self.f_old = np.empty((lx,ly,9))  # 用于放置九个方向速度的分布函数，最后的二个方向的流体速度由这九个速度合成
        self.f_new = np.empty((lx,ly,9))
        # D2Q9格子中每个速度的权重
        self.w = np.array([4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0])
        # 每个速度的方向
        self.e = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])

        self.num_step = kwargs['step_number'] if 'step_number' in kwargs else 60000  # 模拟的步数

    # 平衡态下的流体速度分布函数（(x,y)是格点坐标，i是分速度方向）
    def f_eq(self,x,y,i):
        u_sq = self.vel[x,y,0]**2+self.vel[x,y,1]**2  # 这个格点的宏观流速的平方
        u_proj = np.dot(self.e[i],self.vel[x,y])      # 计算宏观流速与分速度e_i的点积，即宏观流速在e_i方向上的投影
        return self.rho[x,y]*self.w[i]*(1.0+3.0*u_proj+4.5*u_proj**2-1.5*u_sq)

    # 通过碰撞与流动（collision or streaming）更新分布函数，计算下一时刻的流体状态
    def Collision_n_Streaming(self):
        for x, y in self.coord_inner:
            for i in range(9):
                x_pre, y_pre = [x-self.e[i][0],y-self.e[i][1]]                 # 计算碰撞来源的格点坐标
                # 此处，我们采用LBGK单松弛模型（lattice Bhatnagar–Gross–Krook model）处理碰撞，并通过流动更新分布函数
                self.f_new[x,y,i] = (1.0-self.inv_tau)*self.f_old[x_pre,y_pre,i]-self.inv_tau*self.f_eq(x_pre,y_pre,i)

    # 更新宏观流体参数：密度rho, 流速vel等
    def Update_MacroVariable(self):
        for x, y in self.coord_inner:
            self.rho[x,y] = 0.0        # 重置密度
            self.vel[x,y] = [0.0,0.0]  # 重置流速
            for i in range(9):
                self.f_old[x,y,i] = self.f_new[x,y,i]  # 更新分布函数
                self.rho[x,y] += self.f_new[x,y,i]     # 通过求和更新密度（质量守恒定律）
                self.vel[x,y,0] += self.e[i,0]*self.f_new[x,y,i]  # 更新x方向的速度
                self.vel[x,y,1] += self.e[i,1]*self.f_new[x,y,i]  # 更新y方向的速度
            self.vel[x,y] /= self.rho[x,y]  # 除以密度进行归一化（动量守恒定律）

    # 边界条件相关模块
    # 计算所采取的边界条件下，边界上的流速分布函数；boundary: [0,1,2,3] -> [left,top,right,bottom]
    def BoundaryCondition(self,boundary,bc_type,x_bc,y_bc,x_nearby,y_nearby,obstacle='False'):
        if obstacle == 'False':
            if bc_type[boundary] == 0:    # Dirichlet boundary condition
                self.vel[x_bc,y_bc] = self.bc_vel[boundary]
            elif bc_type[boundary] == 1:  # Neumann boundary condition
                self.vel[x_bc,y_bc] = self.vel[x_nearby,y_nearby]
        else:
            pass
        self.rho[x_bc,y_bc] = self.rho[x_nearby, y_nearby]
        for i in range(9):
            self.f_old[x_bc,y_bc,i] = self.f_eq(x_bc,y_bc,i)-self.f_eq(x_nearby,y_nearby,i)+self.f_old[x_nearby,y_nearby,i]  # 更新分布函数

    # 施加边界条件
    def Apply_BoundaryCondition(self):
        # 左右边界（left and right boundaries）
        for n in range(1, self.ly-1):
            self.BoundaryCondition(0,self.bc_type,0,n,1,n)                  # 左边界：x_bc = 0, y_bc = n, x_nearby = 1, y_nearby = n
            self.BoundaryCondition(2,self.bc_type,self.lx-1,n,self.lx-2,n)  # 右边界：x_bc = lx-1, y_bc = n, x_nearby = lx-2, y_nearby = n

        # 上下边界（top and bottom）
        for n in range(self.lx):
            self.BoundaryCondition(1,self.bc_type,n,self.ly-1,n,self.ly-2)  # 顶边界：x_bc = n, y_bc = ly-1, x_nearby = n, y_nearby = ly-2
            self.BoundaryCondition(3,self.bc_type,n,0,n,1)                  # 底边界：x_bc = n, y_bc = 0, x_nearby = n, y_nearby = 1

        # 圆柱体障碍物（cylindrical obstacle）
        for x, y in self.coord:  # Note: for cuda backend, putting 'if statement' inside loops can be much faster!
            if self.cylinder == 'True' and self.obstacle[x, y] == 1:
                self.vel[x, y] = np.array([0.0,0.0])  # velocity is zero within the obstacle and at solid boundary

                x_nearby, y_nearby = [0, 0]  # 初始化障碍物的边界周边条件
                # 此判断配合循环可以描绘出障碍物的形状
                if x >= self.cy_param[0]:
                    x_nearby = x + 1
                else:
                    x_nearby = x - 1
                if y >= self.cy_param[1]:
                    y_nearby = y + 1
                else:
                    y_nearby = y - 1
                self.BoundaryCondition(0, self.bc_type, x, y, x_nearby, y_nearby,obstacle=self.cylinder)

    # 模拟模块（Simulation module）
    def Initialize(self):
        for x, y in self.coord:
            self.vel[x, y] = np.array([0.0,0.0])
            self.rho[x, y] = 1.0
            self.obstacle[x, y] = 0.0
            for i in range(9):
                self.f_new[x, y, i] = self.f_eq(x, y, i)
                self.f_old[x, y, i] = self.f_new[x, y, i]
            if (self.cylinder == 'True'):
                distance = (x-self.cy_param[0])**2.0+(y-self.cy_param[1])**2.0  # 计算坐标与障碍物圆心的距离
                if distance <= self.cy_param[2]:
                    self.obstacle[x, y] = 1.0

    def Simulate(self):
        # gui = ti.GUI('lbm solver', (self.nx, 2 * self.ny))
        self.Initialize()                   # 流体条件初始化
        for i in range(self.num_step):
            self.Collision_n_Streaming()    # 碰撞与流动
            self.Update_MacroVariable()     # 更新宏观参数
            self.Apply_BoundaryCondition()  # 施加边界条件
            ##  code fragment displaying vorticity is contributed by woclass
            vel_x_grad = np.gradient(self.vel[:, :, 0])  # 计算x方向流速的梯度
            vel_y_grad = np.gradient(self.vel[:, :, 1])  # 计算y方向流速的梯度
            vor = vel_x_grad[1]-vel_y_grad[0]            # 计算旋度场
            # print(self.vel[:, :].shape)
            # vel_value= np.linalg.norm(self.vel[:, :],ord=2)  # 计算速率场
            vel_value = (self.vel[:, :, 0] ** 2.0 + self.vel[:, :, 1] ** 2.0) ** 0.5
            print(vel_value.shape)
            ## color map
            colors = [(1, 1, 0), (0.953, 0.490, 0.016), (0, 0, 0),(0.176, 0.976, 0.529), (0, 1, 1)]
            my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                'my_cmap', colors)
            vor_img = cm.ScalarMappable(norm=matplotlib.colors.Normalize(
                vmin=-0.02, vmax=0.02), cmap=my_cmap).to_rgba(vor)
            vel_img = cm.plasma(vel_value)
            # img = np.concatenate((vor_img, vel_img), axis=1)  # 通过将矩阵串接来串接图像
            # plt.set_image(img)
            plt.show()
            if (i % 1000 == 0):
                print('Step: {:}'.format(i))
                # ti.imwrite((img[:,:,0:3]*255).astype(np.uint8), 'fig/karman_'+str(i).zfill(6)+'.png')

    # def pass_to_py(self):
        # return self.vel.to_numpy()[:, :, 0]


if __name__ == '__main__':
    KVS = LBM_KVS()
    KVS.Simulate()