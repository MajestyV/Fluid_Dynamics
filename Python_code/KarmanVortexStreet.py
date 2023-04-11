import numpy as np

class LBM_KVS:
    ''' This code is designed for simulating the Karman vortex street using the lattice Boltzmann method. '''
    def __init__(self,**kwargs):
        # 本代码采用无量纲的流体动力学框架，在此框架下，dx = dy = dt = 1.0（一个lattice units）
        # 设置管道形状
        self.lx = kwargs['length'] if 'length' in kwargs else 801  # 管道的长
        self.ly = kwargs['width'] if 'width' in kwargs else  201   # 管道的宽
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
        self.tau =



        lbm = lbm_solver(801, 201, 0.01, [0, 0, 1, 0],
                         [[0.1, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                         1, [160.0, 100.0, 20.0])

        def __init__(self,
                     nx,  # domain size
                     ny,
                     niu,  # viscosity of fluid
                     bc_type,  # [left,top,right,bottom] boundary conditions: 0 -> Dirichlet ; 1 -> Neumann
                     bc_value,  # if bc_type = 0, we need to specify the velocity in bc_value
                     cy=0,  # whether to place a cylindrical obstacle
                     cy_para=[0.0, 0.0, 0.0],  # location and radius of the cylinder
                     steps=60000):  # total steps to run
            self.niu = niu
            self.tau = 3.0 * niu + 0.5
            self.inv_tau = 1.0 / self.tau
            self.rho = ti.field(dtype=ti.f32, shape=(nx, ny))
            self.vel = ti.Vector.field(2, dtype=ti.f32, shape=(nx, ny))
            self.mask = ti.field(dtype=ti.f32, shape=(nx, ny))
            self.f_old = ti.Vector.field(9, dtype=ti.f32, shape=(nx, ny))
            self.f_new = ti.Vector.field(9, dtype=ti.f32, shape=(nx, ny))
            self.w = ti.field(dtype=ti.f32, shape=9)
            self.e = ti.field(dtype=ti.i32, shape=(9, 2))


            self.steps = steps
            arr = np.array([4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0,
                            1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0], dtype=np.float32)
            self.w.from_numpy(arr)
            arr = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1],
                            [-1, 1], [-1, -1], [1, -1]], dtype=np.int32)
            self.e.from_numpy(arr)

