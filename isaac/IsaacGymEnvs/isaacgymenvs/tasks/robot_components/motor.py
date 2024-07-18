import torch


class Motor:
    def __init__(self):
        print('Motor Init')
        # State Variables
        # self.omega = torch.zeros((m), device=device)     # (rad/s)
        self.motor_supply_voltage = 24                   # V
        
        # Motor Constants (https://www.maxongroup.us/medias/sys_master/root/8816803676190/15-205-EN.pdf)
        self.I = 1.81e-7 
        self.speed_constant = 170 * 2 * torch.pi / 60   # (rad/s)/V
        self.generator_constant = 1/self.speed_constant  # V/(rad/s)
        self.torque_constant = 1/self.speed_constant     # Nm/A
        self.motor_resistance = 0.270                    # Ohms
        self.max_efficiency = 0.92                       # 
        
        self.I_noload = -0.4                            # A
        
        self.I_load   = 0.0                              # A
        self.speed_torque_gradient = 101e3               #rpm/Nm (rpm/mNm in maxon datasheet)
        
        # Useful Variables
        self.rads2rpm = 30/torch.pi
        self.max_current = self.motor_supply_voltage/self.motor_resistance
        self.max_torque = self.max_current*self.torque_constant
        self.motor_scale = 1.0

    # Simulation Required Functions
    def calc_no_load_voltage(self, omega):
        return omega*self.generator_constant
    
    def calc_generator_voltage(self, omega, Il):
        # Il is load current
        return self.calc_no_load_voltage(omega) + self.motor_resistance*Il
    
    def calc_torque(self, Il):
        # Il is load current
        return self.torque_constant * (Il - torch.sign(Il)*self.I_noload)

    def calc_elec_power(self, omega, Il):
        # Il is load current
        # Power is in Watts
        # +P = power out of the battery
        # -P = power into the battery
        return self.calc_generator_voltage(omega, Il) * Il
    
    def calc_mech_power(self, omega, Il):
        # Power is in Watts
        # +P is energy into the system (Creating thrust with the propeller and accellerating the prop)
        # -P is energy out of the system (Charging the battery and decellerating the prop)
        return omega*self.calc_torque(Il)
    
    def get_torque(self, torque_request, omega):
        # Update Omega from thruster sim
        self.I_load_request = torque_request / self.torque_constant
        
        self.generator_voltage = self.calc_generator_voltage(omega, self.I_load) # Using the previous load, calculate the generator induced voltage
        
        max_available_voltage = self.motor_supply_voltage - self.generator_voltage
        max_available_voltage = torch.where(max_available_voltage < 0, torch.zeros_like(max_available_voltage), max_available_voltage)
        max_available_current = max_available_voltage/self.motor_resistance
        
        min_available_voltage = -self.motor_supply_voltage - self.generator_voltage
        min_available_voltage = torch.where(min_available_voltage > 0, torch.zeros_like(min_available_voltage), min_available_voltage)
        min_available_current = min_available_voltage/self.motor_resistance

        self.I_load = (torch.clip(self.I_load_request, min_available_current, max_available_current) + self.I_load)/2
        self.torque_delivered = self.I_load * self.torque_constant

        self.electrical_power = self.calc_elec_power(omega, self.I_load)
        self.mechanical_power = self.calc_mech_power(omega, self.I_load)
        # print('I_REQ {} \nI_MIN{} \nIMAX{} \nI_DEL{}'.format(self.I_load_request, min_available_current, max_available_current, self.I_load))
        return self.torque_delivered
        
    # Additional Functions

    # def calc_max_load_current(self, omega):
    #     return self.calc_no_load_voltage(omega) / self.motor_resistance
    
    # def calc_max_elec_power(self): 
    #     # The efficiency is always slightly below 50% at this operating point; hence, the mechanical input power is roughly twice this value.
    #     I_max = self.calc_max_load_current()/2
    #     U_max = self.calc_generator_voltage(I_max)
    #     P_max = (1/4) * I_max * U_max
    #     P_max2 = (self.omega**2/4) * (1/self.speed_torque_gradient)
    #     #The two above should be the same right?
        
    # def calc_generator_efficiency(self, Il):
    #     return self.calc_elec_power(Il) / self.calc_mech_power(Il)
    
    # def calc_optimum_gen_operation(self):
    #     C = (self.calc_elec_power(Il) * self.I0) / (2(self.motor_resistance))
    #     Il_opt = torch.pow(C, (3/2))
    #     speed_opt = 2*self.motor_resistance*Il_opt * (Il_opt + self.I0)/(self.I0*self.torque_constant)


    

