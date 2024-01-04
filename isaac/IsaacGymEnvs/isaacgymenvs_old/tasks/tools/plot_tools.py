def wing_graph(self, wing_num=0, env_num=0, scale=1):
        num_lines = self.cfg["env"]["glider"]["station_pts"]-1
        line_vertices = torch.zeros((num_lines*2,3), device=self.device, dtype=torch.float)

        line_color = torch.tensor([[1.0, 1.0, 1.0]], device=self.device, dtype=torch.float).repeat(num_lines,1)
        line_vertices[0::2,:] = self.physics.r_global_wnm3[wing_num,0:-1,env_num,:] + self.glider_states[0,0:3]
        line_vertices[1::2,:] = self.physics.r_global_wnm3[wing_num,1:,env_num,:] + self.glider_states[0,0:3]

        self.gym.add_lines(self.viewer, self.envs[0], num_lines, line_vertices.cpu().detach(), line_color.cpu().detach())


def force_graph(self, wing_num=0, env_num=0, scale=1):
    F_global = self.physics.F_global_wnm3.clone()

    num_lines = self.cfg["env"]["glider"]["station_pts"]-1
    line_vertices = torch.zeros((num_lines*2,3), device=self.device, dtype=torch.float)

    line_color = torch.tensor([[1.0, 1.0, 0.0]], device=self.device, dtype=torch.float).repeat(num_lines,1)
    line_vertices[0::2,:] = self.physics.r_global_wnm3[wing_num,0:-1,env_num,:] + scale*F_global[wing_num,0:-1,env_num,:] + self.glider_states[0,0:3]
    line_vertices[1::2,:] = self.physics.r_global_wnm3[wing_num,1:,env_num,:] + scale*F_global[wing_num,1:,env_num,:] + self.glider_states[0,0:3]

    self.gym.add_lines(self.viewer, self.envs[0], num_lines, line_vertices.cpu().detach(), line_color.cpu().detach())


def lift_graph(self, wing_num=0, env_num=0, scale=1):
    L_global = self.physics.Lift_global_wnm3.clone()

    num_lines = self.cfg["env"]["glider"]["station_pts"]-1
    line_vertices = torch.zeros((num_lines*2,3), device=self.device, dtype=torch.float)

    line_color = torch.tensor([[0.0, 1.0, 0.0]], device=self.device, dtype=torch.float).repeat(num_lines,1)
    line_vertices[0::2,:] = self.physics.r_global_wnm3[wing_num,0:-1,env_num,:] + scale*L_global[wing_num,0:-1,env_num,:] + self.glider_states[0,0:3]
    line_vertices[1::2,:] = self.physics.r_global_wnm3[wing_num,1:,env_num,:] + scale*L_global[wing_num,1:,env_num,:] + self.glider_states[0,0:3]

    self.gym.add_lines(self.viewer, self.envs[0], num_lines, line_vertices.cpu().detach(), line_color.cpu().detach())

def drag_graph(self, wing_num=0, env_num=0, scale=1):
    D_global = self.physics.Drag_global_wnm3.clone()
    num_lines = self.cfg["env"]["glider"]["station_pts"]-1
    line_vertices = torch.zeros((num_lines*2,3), device=self.device, dtype=torch.float)

    line_color = torch.tensor([[1.0, 0.0, 0.0]], device=self.device, dtype=torch.float).repeat(num_lines,1)
    line_vertices[0::2,:] = self.physics.r_global_wnm3[wing_num,0:-1,env_num,:] + scale*D_global[wing_num,0:-1,env_num,:] + self.glider_states[0,0:3]
    line_vertices[1::2,:] = self.physics.r_global_wnm3[wing_num,1:,env_num,:] + scale*D_global[wing_num,1:,env_num,:] + self.glider_states[0,0:3]

    self.gym.add_lines(self.viewer, self.envs[0], num_lines, line_vertices.cpu().detach(), line_color.cpu().detach())

def wind_graph(self, vec_scale=1, sym_scale=1, sym_offset=(0.0, 0.0, 0.5)):
    sym_offset = torch.tensor(sym_offset, device=self.device)
    
    num_lines = 1
    line_vertices = torch.zeros((num_lines*2,3), device=self.device, dtype=torch.float)
    line_color = torch.tensor([[0.0, 1.0, 1.0]], device=self.device, dtype=torch.float).repeat(num_lines,1)
    line_vertices[0,:] = torch.zeros_like(self.physics.W_global[0, :, 0], device=self.device) + self.glider_states[0,0:3] + sym_offset
    line_vertices[1,:] = vec_scale*self.physics.WAG[0, :] + self.glider_states[0,0:3] + sym_offset
    
    self.gym.add_lines(self.viewer, self.envs[0], num_lines, line_vertices.cpu().detach(), line_color.cpu().detach())
    
    num_lines = 4 
    line_vertices = torch.zeros((num_lines*2,3), device=self.device, dtype=torch.float)
    line_color = torch.tensor([[0.0, 0.6, 1.0]], device=self.device, dtype=torch.float).repeat(num_lines,1)
    vert_ = torch.tensor([[-sym_scale, 0.0, sym_scale],
                            [sym_scale, 0.0, -sym_scale],
                            [-sym_scale, 0.0, -sym_scale],
                            [sym_scale, 0.0, sym_scale],
                            [-sym_scale, 0.0, sym_scale]] , device=self.device)
    line_vertices[0::2,:] = vert_[0:-1,:] + sym_offset + self.glider_states[0,0:3]
    line_vertices[1::2,:] = vert_[1:,  :] + sym_offset + self.glider_states[0,0:3]

    self.gym.add_lines(self.viewer, self.envs[0], num_lines, line_vertices.cpu().detach(), line_color.cpu().detach())

def destination_graph(self, vec_scale=1, sym_scale=1, sym_offset=(0.0, 0.0, 0.5)):
    sym_offset = torch.tensor(sym_offset, device=self.device)
    
    num_lines = 1
    line_vertices = torch.zeros((num_lines*2,3), device=self.device, dtype=torch.float)
    line_color = torch.tensor([[0.0, 1.0, 1.0]], device=self.device, dtype=torch.float).repeat(num_lines,1)
    line_vertices[0,:] = torch.zeros_like(self.physics.W_global[0, :, 0], device=self.device) + self.glider_states[0,0:3] + sym_offset
    line_vertices[1,:] = vec_scale*self.physics.WAG[0, :] + self.glider_states[0,0:3] + sym_offset
    
    self.gym.add_lines(self.viewer, self.envs[0], num_lines, line_vertices.cpu().detach(), line_color.cpu().detach())
    
    num_lines = 4 
    line_vertices = torch.zeros((num_lines*2,3), device=self.device, dtype=torch.float)
    line_color = torch.tensor([[0.0, 0.6, 1.0]], device=self.device, dtype=torch.float).repeat(num_lines,1)
    vert_ = torch.tensor([[-sym_scale, 0.0, sym_scale],
                            [sym_scale, 0.0, -sym_scale],
                            [-sym_scale, 0.0, -sym_scale],
                            [sym_scale, 0.0, sym_scale],
                            [-sym_scale, 0.0, sym_scale]] , device=self.device)
    line_vertices[0::2,:] = vert_[0:-1,:] + sym_offset + self.glider_states[0,0:3]
    line_vertices[1::2,:] = vert_[1:,  :] + sym_offset + self.glider_states[0,0:3]

    self.gym.add_lines(self.viewer, self.envs[0], num_lines, line_vertices.cpu().detach(), line_color.cpu().detach())

def heading_graph(self, vec_scale=1, sym_scale=1, sym_offset=(0.0, 0.0, 0.25)):
    sym_offset = torch.tensor(sym_offset, device=self.device)
    
    num_lines = 1
    line_vertices = torch.zeros((num_lines*2,3), device=self.device, dtype=torch.float)
    line_color = torch.tensor([[0.0, 1.0, 0.0]], device=self.device, dtype=torch.float).repeat(num_lines,1)
    line_vertices[0,:] = torch.zeros_like(self.physics.W_global[0, :, 0], device=self.device) + self.glider_states[0,0:3] + sym_offset
    north = torch.tensor([1.0, 0.0, 0.0], device=self.device)
    line_vertices[1,:] = vec_scale*north + self.glider_states[0,0:3] + sym_offset
    
    self.gym.add_lines(self.viewer, self.envs[0], num_lines, line_vertices.cpu().detach(), line_color.cpu().detach())
    
    num_lines = 4 
    line_vertices = torch.zeros((num_lines*2,3), device=self.device, dtype=torch.float)
    line_color = torch.tensor([[0.2, 0.1, 0.2]], device=self.device, dtype=torch.float).repeat(num_lines,1)
    vert_ = torch.tensor([[-sym_scale, 0.0, sym_scale],
                            [sym_scale, 0.0, -sym_scale],
                            [-sym_scale, 0.0, -sym_scale],
                            [sym_scale, 0.0, sym_scale],
                            [-sym_scale, 0.0, sym_scale]] , device=self.device)
    line_vertices[0::2,:] = vert_[0:-1,:] + sym_offset + self.glider_states[0,0:3]
    line_vertices[1::2,:] = vert_[1:,  :] + sym_offset + self.glider_states[0,0:3]

    self.gym.add_lines(self.viewer, self.envs[0], num_lines, line_vertices.cpu().detach(), line_color.cpu().detach())


def draw_all_line_graphs(self):
    self.gym.clear_lines(self.viewer)
    self.wind_graph(vec_scale = 1, sym_scale = 0.1)
    self.heading_graph(vec_scale = 1, sym_scale = 0.1)

    self.wing_graph(wing_num=0, env_num=self.env_2_watch, scale=0.1)
    self.wing_graph(wing_num=1, env_num=self.env_2_watch, scale=0.1)
    self.wing_graph(wing_num=2, env_num=self.env_2_watch, scale=0.1)
    # self.wing_graph(wing_num=3, env_num=self.env_2_watch, scale=0.1)

    self.lift_graph(wing_num=0, env_num=self.env_2_watch, scale=0.1)
    self.lift_graph(wing_num=1, env_num=self.env_2_watch, scale=0.1)
    self.lift_graph(wing_num=2, env_num=self.env_2_watch, scale=0.1)
    # self.lift_graph(wing_num=3, env_num=self.env_2_watch, scale=0.1)

    self.drag_graph(wing_num=0, env_num=self.env_2_watch, scale=0.1)
    self.drag_graph(wing_num=1, env_num=self.env_2_watch, scale=0.1)
    self.drag_graph(wing_num=2, env_num=self.env_2_watch, scale=0.1)
    # self.drag_graph(wing_num=3, env_num=self.env_2_watch, scale=0.1)
    
    self.force_graph(wing_num=0, env_num=self.env_2_watch, scale=0.1)
    self.force_graph(wing_num=1, env_num=self.env_2_watch, scale=0.1)
    self.force_graph(wing_num=2, env_num=self.env_2_watch, scale=0.1)
    # self.force_graph(wing_num=3, env_num=self.env_2_watch, scale=0.1)
