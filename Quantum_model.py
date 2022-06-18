"""
Author: Maniraman Periyasamy
Organization: Maniraman Periyasamy

This file constructs different circuits to be used as function approximation. Each of these circuits differe only by the encoding strategy (encoding circuit), whereas the parameterized circuit remains constant.
The list of circuits that can be constructed are as follows:

    1. base : rx encoding with all encoding in the starting (10 rx gates).
    2. localized_encoding_rx : rx encoding with different incremental uploading. 
    3. localized_encoding_deep : rx encoding with different incremental uploading (Deeper variational circuit). 
    4. localized_encoding_multi : rx-ry encoding with different incremental uploading. 
    5. localized_encoding_multi_entang : rx-ry-cz encoding with different incremental uploading. 
    6. full_dataReuploading : rx encoding with full image uploading in front of every variational layer.

Note: Only the parent class is commented as the child class function and arguments are same as parent class.

"""


from tokenize import Exponent
import cirq, sympy
import tensorflow_quantum as tfq
import math

class QcModel:
    """_summary_
    Parent class from which all child class inherits.
    """

    def __init__(self, num_qubits, num_layers, gradient_type='adjoint', data_reuploading=True) -> None:
        """_summary_

        Args:
            num_qubits (int): number of qubits in circuit 
            num_layers (int): number of quantum layers to use
            gradient_type (str, optional): type of tfq gradient to use. Defaults to 'adjoint'.
            data_reuploading (bool, optional): Flag for data re-uploading. Defaults to True.
        """

        self.num_qubits = num_qubits
        self.num_layers = num_layers
        #self.data_type = data_type
        self.qr = cirq.GridQubit.rect(1, num_qubits) 
        
        self.qcm = cirq.Circuit()
        self.qcm_weights = cirq.Circuit()

        if gradient_type == 'adjoint':
            self.gradient = tfq.differentiators.Adjoint()
        elif gradient_type == 'parameter-shift':
            self.gradient = tfq.differentiators.ParameterShift()
        elif gradient_type == 'finite-diff':
            self.gradient = tfq.differentiators.ForwardDifference()
        
        self.weight_parameters = []
        self.encoding_parameters = []

        self.data_reuploading = data_reuploading
        self.psi_counter = 0
        self.x_input_counter = 0
        #self.qiskit_NN_type = qiskit_NN_type
    
    def build_module(self, add_encoding = True):
        """_summary_
        This function builds the quantum circuit
        Args:
            add_encoding (bool, optional): Flag for adding encoding layer before every variational layer. Defaults to True.
        """

        if add_encoding:
            self.encoding_parameters = []

        for i in range(self.num_layers):
            current_layer = cirq.Circuit()
            if add_encoding and (self.data_reuploading or i == 0):
                encoding_circuit_layer, encoding_parameters_layer = self.rx_encoding()
                current_layer = current_layer + encoding_circuit_layer
                self.encoding_parameters = self.encoding_parameters + encoding_parameters_layer
                self.x_input_counter = self.x_input_counter + self.num_qubits
            qc_layer = self.single_layer()
            current_layer = current_layer + qc_layer

            self.qcm = self.qcm + current_layer
    
        observable = self.Z_expectations()
        #print(self.qcm)
        self.QC =  tfq.layers.ControlledPQC(self.qcm, observable, differentiator=self.gradient)
    
    def rx_encoding(self):
        """_summary_
        Type of encoding gate to use.
        Returns:
            encoding_circuit (cirq.circuit): encoding circuit
            parameters (list): list of parameters in this encoding circuit
        """
        encoding_parameters = []
        gate_list = []

        for i in range(self.num_qubits):
            theta = sympy.Symbol('x' + str(self.x_input_counter + i))
            gate_list.append(cirq.rx(theta)(self.qr[i]))
            encoding_parameters.append(theta)

        encoding_circuit = cirq.Circuit(gate_list)
        return encoding_circuit, encoding_parameters 

    def single_layer(self):
        """_summary_
        This function create a single variational layer in the quantum circuit.
        Returns:
            cirq.Circuit: variational circuit.
        """
        gate_list = []
        
        for i in range(self.num_qubits):
            
            psi_y = sympy.Symbol('psi'+str(self.psi_counter))
            self.psi_counter += 1
            psi_z = sympy.Symbol('psi'+str(self.psi_counter))
            self.psi_counter += 1

            gate_list.append(cirq.ry(psi_y)(self.qr[i]))
            gate_list.append(cirq.rz(psi_z)(self.qr[i]))

            self.weight_parameters.extend([psi_y,psi_z])
            
        for i in range(self.num_qubits):
            if i != self.num_qubits-1:
                gate_list.append(cirq.CZ(self.qr[i],self.qr[i+1]))
            else:
                gate_list.append(cirq.CZ(self.qr[i],self.qr[0]))
            
        
        return cirq.Circuit(gate_list)

    def Z_expectations(self):
        """_summary_
        This function implements Z observables to measure the quantum circuit.
        Returns:
            op_list (circ.gate): list of observables for each qubit.
        """
        op_list = []
        for i in range(self.num_qubits):
            op_list.append(cirq.Z(self.qr[i]))	
        return op_list


class base_model(QcModel):
    def __init__(self, num_qubits=10, num_layers=10, gradient_type='adjoint', data_reuploading=True) -> None:
        super().__init__(num_qubits, num_layers, gradient_type=gradient_type, data_reuploading=data_reuploading)

    def single_layer(self):
        gate_list = []
        
        for i in range(self.num_qubits):
            
            psi_y = sympy.Symbol('psi'+str(self.psi_counter))
            self.psi_counter += 1
            psi_z = sympy.Symbol('psi'+str(self.psi_counter))
            self.psi_counter += 1
            
            gate_list.append(cirq.ry(psi_y)(self.qr[i]))
            gate_list.append(cirq.rz(psi_z)(self.qr[i]))
            
            self.weight_parameters.extend([psi_y,psi_z])
            
        for i in range(self.num_qubits):
            if i != self.num_qubits-1:
                gate_list.append(cirq.CZ(self.qr[i],self.qr[i+1]))
            else:
                gate_list.append(cirq.CZ(self.qr[i],self.qr[0]))
        return cirq.Circuit(gate_list)
    
    def build_module(self, add_encoding = True):

        if add_encoding:
            self.encoding_parameters = []
        
        for i in range(self.num_layers):
            current_layer = cirq.Circuit()
            encoding_circuit_layer, encoding_parameters_layer = self.rx_encoding()
            current_layer = current_layer + encoding_circuit_layer
            self.encoding_parameters = self.encoding_parameters + encoding_parameters_layer
            self.x_input_counter = self.x_input_counter + self.num_qubits
            self.qcm = self.qcm + current_layer

        for i in range(self.num_layers):
            current_layer = cirq.Circuit()
            qc_layer = self.single_layer()
            current_layer = current_layer + qc_layer
            self.qcm = self.qcm + current_layer
    
        observable = self.Z_expectations()
        #print(self.qcm)
        self.QC =  tfq.layers.ControlledPQC(self.qcm, observable, differentiator=self.gradient)


class localized_encoding_rx(QcModel):
    def __init__(self, num_qubits=10, num_layers=10, gradient_type='adjoint', data_reuploading=True, enc_layers=10) -> None:
        super().__init__(num_qubits, num_layers, gradient_type=gradient_type, data_reuploading=data_reuploading)
        self.enc_layers = enc_layers
    def single_layer(self):
        
        self.total_parameters = len(str(2*self.num_layers*self.num_qubits))
        gate_list = []
        for i in range(self.num_qubits):
            
            psi_y = sympy.Symbol('psi'+str(self.psi_counter).zfill(self.total_parameters))
            self.psi_counter += 1
            psi_z = sympy.Symbol('psi'+str(self.psi_counter).zfill(self.total_parameters))
            self.psi_counter += 1
            
            gate_list.append(cirq.ry(psi_y)(self.qr[i]))
            gate_list.append(cirq.rz(psi_z)(self.qr[i]))
            
            self.weight_parameters.extend([psi_y,psi_z])
            
        for i in range(self.num_qubits):
            if i != self.num_qubits-1:
                gate_list.append(cirq.CZ(self.qr[i],self.qr[i+1]))
            else:
                gate_list.append(cirq.CZ(self.qr[i],self.qr[0]))
        return cirq.Circuit(gate_list)
    
    def encoding_local(self, num_list, add_entangling = False):
        self.total_inputs = len(str(self.num_layers*self.num_qubits))
        encoding_parameters = []
        gate_list = []

        for j in range(num_list):
            for i in range(self.num_qubits):
                theta = sympy.Symbol('x' + str(self.x_input_counter + j*self.num_qubits + i).zfill(self.total_inputs))
                gate_list.append(cirq.rx(theta)(self.qr[i]))
                encoding_parameters.append(theta)
            if add_entangling:
                for i in range(self.num_qubits):
                    if i != self.num_qubits-1:
                        gate_list.append(cirq.CZ(self.qr[i],self.qr[i+1]))
                    else:
                        gate_list.append(cirq.CZ(self.qr[i],self.qr[0]))
                #encoding_parameters.append(theta)

        encoding_circuit = cirq.Circuit(gate_list)
        return encoding_circuit, encoding_parameters 

    def build_module(self, add_encoding = True):

        en_ly_count = []
        lr_count = self.num_layers
        for i in range(self.enc_layers,0,-1):
            in_lr_cnt = math.ceil(lr_count/i)
            en_ly_count.append(in_lr_cnt)
            lr_count = lr_count-in_lr_cnt

        if add_encoding:
            self.encoding_parameters = []
        

        encoding_count = 0

        for i in range(self.num_layers):
            current_layer = cirq.Circuit()
            if encoding_count < len(en_ly_count):
                encoding_circuit_layer, encoding_parameters_layer = self.encoding_local(num_list=en_ly_count[i])
                current_layer = current_layer + encoding_circuit_layer
                self.encoding_parameters = self.encoding_parameters + encoding_parameters_layer
                self.x_input_counter = self.x_input_counter + len(encoding_parameters_layer)
                encoding_count += 1
            qc_layer = self.single_layer()
            current_layer = current_layer + qc_layer
            self.qcm = self.qcm + current_layer
    
        observable = self.Z_expectations()
        #print(self.qcm)
        self.QC =  tfq.layers.ControlledPQC(self.qcm, observable, differentiator=self.gradient)


class localized_encoding_multi(QcModel):


    def __init__(self, num_qubits=10, num_layers=10, gradient_type='adjoint', data_reuploading=False, enc_layers=10) -> None:
        super().__init__(num_qubits, num_layers, gradient_type=gradient_type, data_reuploading=data_reuploading)
        self.enc_layers = enc_layers
        self.toggle_rx = True
    def single_layer(self):
        
        self.total_parameters = len(str(2*self.num_layers*self.num_qubits))
        gate_list = []
        for i in range(self.num_qubits):
            
            psi_y = sympy.Symbol('psi'+str(self.psi_counter).zfill(self.total_parameters))
            self.psi_counter += 1
            psi_z = sympy.Symbol('psi'+str(self.psi_counter).zfill(self.total_parameters))
            self.psi_counter += 1
            
            gate_list.append(cirq.ry(psi_y)(self.qr[i]))
            gate_list.append(cirq.rz(psi_z)(self.qr[i]))
            
            self.weight_parameters.extend([psi_y,psi_z])
            
        for i in range(self.num_qubits):
            if i != self.num_qubits-1:
                gate_list.append(cirq.CZ(self.qr[i],self.qr[i+1]))
            else:
                gate_list.append(cirq.CZ(self.qr[i],self.qr[0]))
        return cirq.Circuit(gate_list)
    
    def encoding_local(self, num_list, add_entangling = False):
        if self.data_reuploading == False:
            self.total_inputs = len(str(self.num_layers*self.num_qubits))
        else:
            self.total_inputs = len(str(self.num_layers*self.num_qubits*10))
        encoding_parameters = []
        gate_list = []
        

        for j in range(num_list):
            for i in range(self.num_qubits):
                theta = sympy.Symbol('x' + str(self.x_input_counter + j*self.num_qubits + i).zfill(self.total_inputs))
                if self.toggle_rx:
                    gate_list.append(cirq.rx(theta)(self.qr[i]))
                else:
                    gate_list.append(cirq.ry(theta)(self.qr[i]))
                encoding_parameters.append(theta)
            if add_entangling:
                for i in range(self.num_qubits):
                    if i != self.num_qubits-1:
                        gate_list.append(cirq.CZ(self.qr[i],self.qr[i+1]))
                    else:
                        gate_list.append(cirq.CZ(self.qr[i],self.qr[0]))
            self.toggle_rx = not self.toggle_rx
                #encoding_parameters.append(theta)

        encoding_circuit = cirq.Circuit(gate_list)
        return encoding_circuit, encoding_parameters 

    def build_module(self, add_encoding = True):

        if self.data_reuploading == False:
            en_ly_count = []
            lr_count = self.num_layers
            for i in range(self.enc_layers,0,-1):
                in_lr_cnt = math.ceil(lr_count/i)
                en_ly_count.append(in_lr_cnt)
                lr_count = lr_count-in_lr_cnt
        else:
            en_ly_count = [10,10,10,10,10,10,10,10,10,10]

        if add_encoding:
            self.encoding_parameters = []
        

        encoding_count = 0

        for i in range(self.num_layers):
            current_layer = cirq.Circuit()
            if encoding_count < len(en_ly_count):
                encoding_circuit_layer, encoding_parameters_layer = self.encoding_local(num_list=en_ly_count[i])
                current_layer = current_layer + encoding_circuit_layer
                self.encoding_parameters = self.encoding_parameters + encoding_parameters_layer
                self.x_input_counter = self.x_input_counter + len(encoding_parameters_layer)
                encoding_count += 1
            qc_layer = self.single_layer()
            current_layer = current_layer + qc_layer
            self.qcm = self.qcm + current_layer
    
        observable = self.Z_expectations()
        #print(self.qcm)
        self.QC =  tfq.layers.ControlledPQC(self.qcm, observable, differentiator=self.gradient)

class localized_encoding_multi_entang(QcModel):

    def __init__(self, num_qubits=10, num_layers=10, gradient_type='adjoint', data_reuploading=False, enc_layers=10) -> None:
        super().__init__(num_qubits, num_layers, gradient_type=gradient_type, data_reuploading=data_reuploading)
        self.enc_layers = enc_layers
        self.toggle_rx = True
    def single_layer(self):
        
        self.total_parameters = len(str(2*self.num_layers*self.num_qubits))
        gate_list = []
        for i in range(self.num_qubits):
            
            psi_y = sympy.Symbol('psi'+str(self.psi_counter).zfill(self.total_parameters))
            self.psi_counter += 1
            psi_z = sympy.Symbol('psi'+str(self.psi_counter).zfill(self.total_parameters))
            self.psi_counter += 1
            
            gate_list.append(cirq.ry(psi_y)(self.qr[i]))
            gate_list.append(cirq.rz(psi_z)(self.qr[i]))
            
            self.weight_parameters.extend([psi_y,psi_z])
            
        for i in range(self.num_qubits):
            if i != self.num_qubits-1:
                gate_list.append(cirq.CZ(self.qr[i],self.qr[i+1]))
            else:
                gate_list.append(cirq.CZ(self.qr[i],self.qr[0]))
        return cirq.Circuit(gate_list)
    
    def encoding_local(self, num_list, add_entangling = True):
        if self.data_reuploading == False:
            self.total_inputs = len(str(self.num_layers*self.num_qubits))
        else:
            self.total_inputs = len(str(self.num_layers*self.num_qubits*10))
        encoding_parameters = []
        gate_list = []
        

        for j in range(num_list):
            for i in range(self.num_qubits):
                theta = sympy.Symbol('x' + str(self.x_input_counter + j*self.num_qubits + i).zfill(self.total_inputs))
                if self.toggle_rx:
                    gate_list.append(cirq.rx(theta)(self.qr[i]))
                else:
                    gate_list.append(cirq.ry(theta)(self.qr[i]))
                encoding_parameters.append(theta)
            if add_entangling and self.toggle_rx:
                for i in range(self.num_qubits):
                    if i != self.num_qubits-1:
                        gate_list.append(cirq.CZ(self.qr[i],self.qr[i+1]))
                    else:
                        gate_list.append(cirq.CZ(self.qr[i],self.qr[0]))
            self.toggle_rx = not self.toggle_rx
                #encoding_parameters.append(theta)

        encoding_circuit = cirq.Circuit(gate_list)
        return encoding_circuit, encoding_parameters 

    def build_module(self, add_encoding = True):

        if self.data_reuploading == False:
            en_ly_count = []
            lr_count = self.num_layers
            for i in range(self.enc_layers,0,-1):
                in_lr_cnt = math.ceil(lr_count/i)
                en_ly_count.append(in_lr_cnt)
                lr_count = lr_count-in_lr_cnt
        else:
            en_ly_count = [10,10,10,10,10,10,10,10,10,10]


        if add_encoding:
            self.encoding_parameters = []
        

        encoding_count = 0

        for i in range(self.num_layers):
            current_layer = cirq.Circuit()
            if encoding_count < len(en_ly_count):
                encoding_circuit_layer, encoding_parameters_layer = self.encoding_local(num_list=en_ly_count[i])
                current_layer = current_layer + encoding_circuit_layer
                self.encoding_parameters = self.encoding_parameters + encoding_parameters_layer
                self.x_input_counter = self.x_input_counter + len(encoding_parameters_layer)
                encoding_count += 1
            qc_layer = self.single_layer()
            current_layer = current_layer + qc_layer
            self.qcm = self.qcm + current_layer
    
        observable = self.Z_expectations()
        #print(self.qcm)
        self.QC =  tfq.layers.ControlledPQC(self.qcm, observable, differentiator=self.gradient)

class full_dataReuploading(QcModel):
    def __init__(self, num_qubits=10, num_layers=10, gradient_type='adjoint', data_reuploading=True, enc_layers=10) -> None:
        super().__init__(num_qubits, num_layers, gradient_type=gradient_type, data_reuploading=data_reuploading)
        self.enc_layers = enc_layers
    def single_layer(self):
        
        self.total_parameters = len(str(2*self.num_layers*self.num_qubits))
        gate_list = []
        for i in range(self.num_qubits):
            
            psi_y = sympy.Symbol('psi'+str(self.psi_counter).zfill(self.total_parameters))
            self.psi_counter += 1
            psi_z = sympy.Symbol('psi'+str(self.psi_counter).zfill(self.total_parameters))
            self.psi_counter += 1
            
            gate_list.append(cirq.ry(psi_y)(self.qr[i]))
            gate_list.append(cirq.rz(psi_z)(self.qr[i]))
            
            self.weight_parameters.extend([psi_y,psi_z])
            
        for i in range(self.num_qubits):
            if i != self.num_qubits-1:
                gate_list.append(cirq.CZ(self.qr[i],self.qr[i+1]))
            else:
                gate_list.append(cirq.CZ(self.qr[i],self.qr[0]))
        return cirq.Circuit(gate_list)
    
    def encoding_local(self, num_list, add_entangling = False):
        self.total_inputs = len(str(self.num_layers*self.num_qubits))
        encoding_parameters = []
        gate_list = []

        for j in range(num_list):
            for i in range(self.num_qubits):
                theta = sympy.Symbol('x' + str(self.x_input_counter + j*self.num_qubits + i).zfill(self.total_inputs))
                gate_list.append(cirq.rx(theta)(self.qr[i]))
                encoding_parameters.append(theta)
            if add_entangling:
                for i in range(self.num_qubits):
                    if i != self.num_qubits-1:
                        gate_list.append(cirq.CZ(self.qr[i],self.qr[i+1]))
                    else:
                        gate_list.append(cirq.CZ(self.qr[i],self.qr[0]))
                encoding_parameters.append(theta)

        encoding_circuit = cirq.Circuit(gate_list)
        return encoding_circuit, encoding_parameters 

    def build_module(self, add_encoding = True):

        """if self.enc_layers == 1:
            en_ly_count = [10]
        elif self.enc_layers == 2:
            en_ly_count = [5,5]
        elif self.enc_layers == 4:
            en_ly_count = [3,3,2,2]
        elif self.enc_layers == 6:
            en_ly_count = [2,2,2,2,1,1]
        elif self.enc_layers == 8:
            en_ly_count = [1,2,1,2,1,1,1,1]
        elif self.enc_layers == 10:
            en_ly_count = [1,1,1,1,1,1,1,1,1,1]
        elif self.enc_layers == 100:
            en_ly_count = [10,10,10,10,10,10,10,10,10,10]"""


        en_ly_count = []
        lr_count = self.num_layers*self.num_qubits
        for i in range(self.enc_layers,0,-1):
            in_lr_cnt = math.ceil(lr_count/i)
            en_ly_count.append(in_lr_cnt)
            lr_count = lr_count-in_lr_cnt

        if add_encoding:
            self.encoding_parameters = []
        

        encoding_count = 0

        for i in range(self.num_layers):
            current_layer = cirq.Circuit()
            if encoding_count < len(en_ly_count):
                encoding_circuit_layer, encoding_parameters_layer = self.encoding_local(num_list=en_ly_count[i])
                current_layer = current_layer + encoding_circuit_layer
                self.encoding_parameters = self.encoding_parameters + encoding_parameters_layer
                self.x_input_counter = self.x_input_counter + len(encoding_parameters_layer)
                encoding_count += 1
            qc_layer = self.single_layer()
            current_layer = current_layer + qc_layer
            self.qcm = self.qcm + current_layer
            #qc_layer = self.single_layer()
            #current_layer = current_layer + qc_layer
            #self.qcm = self.qcm + current_layer

        
    
        observable = self.Z_expectations()
        #print(self.qcm)
        self.QC =  tfq.layers.ControlledPQC(self.qcm, observable, differentiator=self.gradient)


class localized_encoding_deep(QcModel):
    def __init__(self, num_qubits=10, num_layers=10, gradient_type='adjoint', data_reuploading=True, enc_layers=10) -> None:
        super().__init__(num_qubits, num_layers, gradient_type=gradient_type, data_reuploading=data_reuploading)
        self.enc_layers = enc_layers
    def single_layer(self):
        
        self.total_parameters = len(str(2*self.num_layers*self.num_qubits))
        gate_list = []
        for i in range(self.num_qubits):
            
            psi_y = sympy.Symbol('psi'+str(self.psi_counter).zfill(self.total_parameters))
            self.psi_counter += 1
            psi_z = sympy.Symbol('psi'+str(self.psi_counter).zfill(self.total_parameters))
            self.psi_counter += 1
            
            gate_list.append(cirq.ry(psi_y)(self.qr[i]))
            gate_list.append(cirq.rz(psi_z)(self.qr[i]))
            
            self.weight_parameters.extend([psi_y,psi_z])
            
        for i in range(self.num_qubits):
            if i != self.num_qubits-1:
                gate_list.append(cirq.CZ(self.qr[i],self.qr[i+1]))
            else:
                gate_list.append(cirq.CZ(self.qr[i],self.qr[0]))
        return cirq.Circuit(gate_list)
    
    def encoding_local(self, num_list, add_entangling = False):
        self.total_inputs = len(str(self.num_layers*self.num_qubits))
        encoding_parameters = []
        gate_list = []

        for j in range(num_list):
            for i in range(self.num_qubits):
                theta = sympy.Symbol('x' + str(self.x_input_counter + j*self.num_qubits + i).zfill(self.total_inputs))
                gate_list.append(cirq.rx(theta)(self.qr[i]))
                encoding_parameters.append(theta)
            if add_entangling:
                for i in range(self.num_qubits):
                    if i != self.num_qubits-1:
                        gate_list.append(cirq.CZ(self.qr[i],self.qr[i+1]))
                    else:
                        gate_list.append(cirq.CZ(self.qr[i],self.qr[0]))
                encoding_parameters.append(theta)

        encoding_circuit = cirq.Circuit(gate_list)
        return encoding_circuit, encoding_parameters 

    def build_module(self, add_encoding = True):

        en_ly_count = []
        lr_count = self.num_layers
        for i in range(self.enc_layers,0,-1):
            in_lr_cnt = math.ceil(lr_count/i)
            en_ly_count.append(in_lr_cnt)
            lr_count = lr_count-in_lr_cnt

        if add_encoding:
            self.encoding_parameters = []
        

        encoding_count = 0

        for i in range(self.num_layers):
            current_layer = cirq.Circuit()
            if encoding_count < len(en_ly_count):
                encoding_circuit_layer, encoding_parameters_layer = self.encoding_local(num_list=en_ly_count[i])
                current_layer = current_layer + encoding_circuit_layer
                self.encoding_parameters = self.encoding_parameters + encoding_parameters_layer
                self.x_input_counter = self.x_input_counter + len(encoding_parameters_layer)
                encoding_count += 1
            qc_layer = self.single_layer()
            current_layer = current_layer + qc_layer
            self.qcm = self.qcm + current_layer
            qc_layer = self.single_layer()
            current_layer = current_layer + qc_layer
            self.qcm = self.qcm + current_layer
            qc_layer = self.single_layer()
            current_layer = current_layer + qc_layer
            self.qcm = self.qcm + current_layer
    
        observable = self.Z_expectations()
        #print(self.qcm)
        self.QC =  tfq.layers.ControlledPQC(self.qcm, observable, differentiator=self.gradient)


def load_architecture(args):

    if args.arch == 'base':
        QcN = base_model(num_qubits=args.num_qubits, num_layers=args.num_layers, data_reuploading=True)
    elif args.arch == 'localized_encoding_rx':
        QcN = localized_encoding_rx(num_qubits=args.num_qubits, num_layers=args.num_layers, data_reuploading=True, enc_layers = args.enc_layers)
    elif args.arch == 'localized_encoding_deep':
        QcN = localized_encoding_deep(num_qubits=args.num_qubits, num_layers=args.num_layers, data_reuploading=True, enc_layers = args.enc_layers)
    elif args.arch == 'localized_encoding_multi':
        QcN = localized_encoding_multi(num_qubits=args.num_qubits, num_layers=args.num_layers, data_reuploading=args.data_reuploading, enc_layers = args.enc_layers)
    elif args.arch == 'localized_encoding_multi_entang':
        QcN = localized_encoding_multi_entang(num_qubits=args.num_qubits, num_layers=args.num_layers, data_reuploading=args.data_reuploading, enc_layers = args.enc_layers)
    elif args.arch == 'full_dataReuploading':
        QcN = full_dataReuploading(num_qubits=args.num_qubits, num_layers=args.num_layers, data_reuploading=True, enc_layers = args.num_layers)
    else:
        QcN = QcModel(num_qubits=args.num_qubits, num_layers=args.num_layers, data_reuploading=True)

    return QcN