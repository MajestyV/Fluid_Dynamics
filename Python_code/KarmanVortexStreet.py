import numpy as np

class LBM_KVS:
    ''' This code is designed for simulating the Karman vortex street using the lattice Boltzmann method. '''
    def __init__(self,**kwargs):
        # 本代码采用无量纲的流体动力学框架，在此框架下，dx = dy = dt = 1.0（一个lattice units）
        # 设置管道形状
        lx = kwargs['length'] if 'length' in kwargs else 801  # 管道的长
        ly = kwargs['width'] if 'width' in kwargs else  201   # 管道的宽
        self.lx, self.ly = (lx, ly)                           # 转换成实例变量
        # 设置管道边界条件：[left,top,right,bottom] boundary conditions: 0 -> Dirichlet ; 1 -> Neumann
        boundary_type = kwargs['boundary_condition'] if 'boundary_condition' in kwargs else [0, 0, 1, 0]
        # 设置边界速度，此代码模拟的是二维的管道（区域）中的流体，所以速度具有两个自由度，即流体速度可分解为沿x方向的Vx和沿y方向的Vy
        boundary_vel = kwargs['boundary_initial'] if 'boundary_initial' in kwargs else [[0.1, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        self.bc_type = np.array(boundary_type)      # 转换为数组，方便后续计算
        self.bc_vel = np.array([boundary_vel])      # 转换为二维数组（亦可理解为二阶张量），shape应为(4,2)
        # 设置障碍物位置大小参数
        self.cylinder = kwargs['cylinder'] if 'cylinder' in kwargs else 'True'  # 选择是否要放置圆柱体障碍物
        cylinder_param = kwargs['cylinder_parameter'] if 'cylinder_parameter' in kwargs else [160.0, 100.0, 20.0]  # 圆柱体参数：(x轴坐标, y轴坐标, 半径)
        self.cy_param = np.array(cylinder_param)    # 转换为数组，方便计算

        # 设置无量纲流体动力学参数
        self.nu = kwargs['viscosity'] if 'viscosity' in kwargs else 0.01  # viscosity of fluid (粘滞系数)
        self.tau = 3.0*self.nu+0.5                                        # 弛豫时间

        # 建立二维九速四方格子的格点玻尔兹曼模型（D2Q9 LBM）
        self.mask = np.empty((lx,ly))
        self.rho = np.empty((lx,ly))      # 创建一个lx行，ly列的二阶张量以存放每个格点上的流体密度
        self.vel = np.empty((lx,ly,2))    # 创建一个形状为 (lx,ly,2) 的三阶张量来存放每个格点上的流体速度（对于二维系统，流体速度有两个自由度）
        self.f_old = np.empty((lx,ly,9))  # 用于放置九个方向速度的分布函数，最后的二个方向的流体速度由这九个速度合成
        self.f_new = np.empty((lx,ly,9))
        # D2Q9格子中每个速度的权重
        self.w = np.array([4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0])
        # 每个速度的方向
        self.e = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])

    # 平衡态下的流体速度分布函数（(x,y)是格点坐标，i是分速度方向）
    def f_eq(self,x,y,i):
        u_sq = self.vel[x,y,0]**2+self.vel[x,y,1]**2  # 这个格点的宏观流速的平方


    @ti.func  # compute equilibrium distribution function
    def f_eq(self, i, j, k):
        eu = ti.cast(self.e[k, 0], ti.f32) * self.vel[i, j][0] + ti.cast(self.e[k, 1],
                                                                         ti.f32) * self.vel[i, j][1]
        uv = self.vel[i, j][0] ** 2.0 + self.vel[i, j][1] ** 2.0
        return self.w[k] * self.rho[i, j] * (1.0 + 3.0 * eu + 4.5 * eu ** 2 - 1.5 * uv)

    @ti.kernel
    def init(self):
        for i, j in self.rho:
            self.vel[i, j][0] = 0.0
            self.vel[i, j][1] = 0.0
            self.rho[i, j] = 1.0
            self.mask[i, j] = 0.0
            for k in ti.static(range(9)):
                self.f_new[i, j][k] = self.f_eq(i, j, k)
                self.f_old[i, j][k] = self.f_new[i, j][k]
            if (self.cy == 1):
                if ((ti.cast(i, ti.f32) - self.cy_para[0]) ** 2.0 + (ti.cast(j, ti.f32)
                                                                     - self.cy_para[1]) ** 2.0 <= self.cy_para[
                    2] ** 2.0):
                    self.mask[i, j] = 1.0

    @ti.kernel
    def collide_and_stream(self):  # lbm core equation
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            for k in ti.static(range(9)):
                ip = i - self.e[k, 0]
                jp = j - self.e[k, 1]
                self.f_new[i, j][k] = (1.0 - self.inv_tau) * self.f_old[ip, jp][k] + \
                                      self.f_eq(ip, jp, k) * self.inv_tau

    @ti.kernel
    def update_macro_var(self):  # compute rho u v
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            self.rho[i, j] = 0.0
            self.vel[i, j][0] = 0.0
            self.vel[i, j][1] = 0.0
            for k in ti.static(range(9)):
                self.f_old[i, j][k] = self.f_new[i, j][k]
                self.rho[i, j] += self.f_new[i, j][k]
                self.vel[i, j][0] += (ti.cast(self.e[k, 0], ti.f32) *
                                      self.f_new[i, j][k])
                self.vel[i, j][1] += (ti.cast(self.e[k, 1], ti.f32) *
                                      self.f_new[i, j][k])
            self.vel[i, j][0] /= self.rho[i, j]
            self.vel[i, j][1] /= self.rho[i, j]

    @ti.kernel
    def apply_bc(self):  # impose boundary conditions
        # left and right
        for j in ti.ndrange(1, self.ny - 1):
            # left: dr = 0; ibc = 0; jbc = j; inb = 1; jnb = j
            self.apply_bc_core(1, 0, 0, j, 1, j)

            # right: dr = 2; ibc = nx-1; jbc = j; inb = nx-2; jnb = j
            self.apply_bc_core(1, 2, self.nx - 1, j, self.nx - 2, j)

        # top and bottom
        for i in ti.ndrange(self.nx):
            # top: dr = 1; ibc = i; jbc = ny-1; inb = i; jnb = ny-2
            self.apply_bc_core(1, 1, i, self.ny - 1, i, self.ny - 2)

            # bottom: dr = 3; ibc = i; jbc = 0; inb = i; jnb = 1
            self.apply_bc_core(1, 3, i, 0, i, 1)

        # cylindrical obstacle
        # Note: for cuda backend, putting 'if statement' inside loops can be much faster!
        for i, j in ti.ndrange(self.nx, self.ny):
            if (self.cy == 1 and self.mask[i, j] == 1):
                self.vel[i, j][0] = 0.0  # velocity is zero at solid boundary
                self.vel[i, j][1] = 0.0
                inb = 0
                jnb = 0
                if (ti.cast(i, ti.f32) >= self.cy_para[0]):
                    inb = i + 1
                else:
                    inb = i - 1
                if (ti.cast(j, ti.f32) >= self.cy_para[1]):
                    jnb = j + 1
                else:
                    jnb = j - 1
                self.apply_bc_core(0, 0, i, j, inb, jnb)

    @ti.func
    def apply_bc_core(self, outer, dr, ibc, jbc, inb, jnb):
        if (outer == 1):  # handle outer boundary
            if (self.bc_type[dr] == 0):
                self.vel[ibc, jbc][0] = self.bc_value[dr, 0]
                self.vel[ibc, jbc][1] = self.bc_value[dr, 1]
            elif (self.bc_type[dr] == 1):
                self.vel[ibc, jbc][0] = self.vel[inb, jnb][0]
                self.vel[ibc, jbc][1] = self.vel[inb, jnb][1]
        self.rho[ibc, jbc] = self.rho[inb, jnb]
        for k in ti.static(range(9)):
            self.f_old[ibc, jbc][k] = self.f_eq(ibc, jbc, k) - self.f_eq(inb, jnb, k) + \
                                      self.f_old[inb, jnb][k]

    def solve(self):
        gui = ti.GUI('lbm solver', (self.nx, 2 * self.ny))
        self.init()
        for i in range(self.steps):
            self.collide_and_stream()
            self.update_macro_var()
            self.apply_bc()
            ##  code fragment displaying vorticity is contributed by woclass
            vel = self.vel.to_numpy()
            ugrad = np.gradient(vel[:, :, 0])
            vgrad = np.gradient(vel[:, :, 1])
            vor = ugrad[1] - vgrad[0]
            vel_mag = (vel[:, :, 0] ** 2.0 + vel[:, :, 1] ** 2.0) ** 0.5
            ## color map
            colors = [(1, 1, 0), (0.953, 0.490, 0.016), (0, 0, 0),
                      (0.176, 0.976, 0.529), (0, 1, 1)]
            my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                'my_cmap', colors)
            vor_img = cm.ScalarMappable(norm=matplotlib.colors.Normalize(
                vmin=-0.02, vmax=0.02), cmap=my_cmap).to_rgba(vor)
            vel_img = cm.plasma(vel_mag / 0.15)
            img = np.concatenate((vor_img, vel_img), axis=1)
            gui.set_image(img)
            gui.show()
            if (i % 1000 == 0):
                print('Step: {:}'.format(i))
                # ti.imwrite((img[:,:,0:3]*255).astype(np.uint8), 'fig/karman_'+str(i).zfill(6)+'.png')

    def pass_to_py(self):
        return self.vel.to_numpy()[:, :, 0]


if __name__ == '__main__':
    flow_case = 0
    if (flow_case == 0):  # von Karman vortex street: Re = U*D/niu = 200
        lbm = lbm_solver(801, 201, 0.01, [0, 0, 1, 0],
                         [[0.1, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                         1, [160.0, 100.0, 20.0])
        lbm.solve()
    elif (flow_case == 1):  # lid-driven cavity flow: Re = U*L/niu = 1000
        lbm = lbm_solver(256, 256, 0.0255, [0, 0, 0, 0],
                         [[0.0, 0.0], [0.1, 0.0], [0.0, 0.0], [0.0, 0.0]])
        lbm.solve()
        # compare with literature results
        y_ref, u_ref = np.loadtxt('data/ghia1982.dat', unpack=True, skiprows=2, usecols=(0, 2))
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 3), dpi=200)
        axes.plot(np.linspace(0, 1.0, 256), lbm.pass_to_py()[256 // 2, :] / 0.1, 'b-', label='LBM')
        axes.plot(y_ref, u_ref, 'rs', label='Ghia et al. 1982')
        axes.legend()
        axes.set_xlabel(r'Y')
        axes.set_ylabel(r'U')
        plt.tight_layout()
        plt.show()