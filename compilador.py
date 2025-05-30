import math

class RiscVSimulator:
    """Simulador simplificado de RISC-V que soporta un subconjunto básico de instrucciones."""
    
    def __init__(self):
        # Registros: x0-x31 (x0 siempre es 0)
        self.registers = [0] * 64
        # Memoria (simulada como un diccionario para acceso eficiente)
        self.memory = {}
        self.data_labels={}
        # Contador de programa
        self.pc = 0
        # Diccionario de instrucciones
        self.instructions = {}
        # Flag para controlar ejecución
        self.running = False

    def load_data_label(self, label, string, base_address=0x10010000):
        """Agrega una etiqueta con .asciiz a la memoria simulada"""
        address = base_address + len(self.data_labels) * 0x20  # espacio reservado por etiqueta
        self.data_labels[label] = address
        for i, char in enumerate(string):
            if char == '\n':
                self.memory[address + i] = 0x0A
            else:
                self.memory[address + i] = ord(char) # guardar cada carácter como byte
        self.memory[address + len(string)] = 0  # Null terminator (\0)
    
    def execute_la(self, rd, label):
        """Simula la instrucción 'la rd, label'"""
        if label not in self.data_labels:
            raise ValueError(f"Etiqueta no encontrada: {label}")
        self.registers[rd] = self.data_labels[label]

    def load_program(self, program):
        """Carga un programa en la memoria del simulador."""
        # Limpiamos el estado previo
        self.registers = [0] * 64
        self.pc = 0
        self.instructions = {}
        
        # Procesamos el programa línea por línea
        address = 0
        for line in program.strip().split('\n'):
            # Ignoramos líneas vacías y comentarios
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Procesamos etiquetas
            if ':' in line:
                label, instruction = line.split(':', 1)
                label = label.strip()
                self.memory[label] = address
                line = instruction.strip()
                if not line:  # Si solo era una etiqueta, continuamos
                    continue
            
            # Guardamos la instrucción
            self.instructions[address] = line
            address += 4  # Cada instrucción ocupa 4 bytes en RISC-V
    
    def run(self):
        """Ejecuta el programa cargado."""
        self.running = True
        self.pc = 0
        
        while self.running and self.pc in self.instructions:
            instruction = self.instructions[self.pc]
            self.execute_instruction(instruction)
        return self.registers[10]  # Devuelve el valor en x10 (a0) como resultado
    
    def execute_instruction(self, instruction):
        """Ejecuta una instrucción individual."""
        parts = instruction.split()
        opcode = parts[0].lower()
        
        # Incrementamos el PC por defecto (algunas instrucciones lo modificarán)
        next_pc = self.pc + 4
        
        # Instrucciones aritméticas
        if opcode == 'add':
            rd, rs1, rs2 = self._parse_r_type(parts[1:])
            self.registers[rd] = self.registers[rs1] + self.registers[rs2]
        
        elif opcode == 'sub':
            rd, rs1, rs2 = self._parse_r_type(parts[1:])
            self.registers[rd] = self.registers[rs1] - self.registers[rs2]
        
        elif opcode == 'addi':
            rd, rs1, imm = self._parse_i_type(parts[1:])
            self.registers[rd] = self.registers[rs1] + imm
        
        elif opcode == 'mul':
            rd, rs1, rs2 = self._parse_r_type(parts[1:])
            self.registers[rd] = self.registers[rs1] * self.registers[rs2]

        elif opcode == 'muli':
            rd, rs1, imm = self._parse_i_type(parts[1:])
            self.registers[rd] = self.registers[rs1] * imm
        
        elif opcode == 'div':
            rd, rs1, rs2 = self._parse_r_type(parts[1:])
            if(self.registers[rs1]==0):
                print("Error, No se puede dividir el numero 0")
            else:
                self.registers[rd] = self.registers[rs1] / self.registers[rs2]

        elif opcode == 'fadd':
            rd, rs1, rs2 = self._parse_r_type(parts[1:])
            self.registers[rd] = float(self.registers[rs1]) + float(self.registers[rs2])

        elif opcode == 'fsub':
            rd, rs1, rs2 = self._parse_r_type(parts[1:])
            self.registers[rd] = float(self.registers[rs1]) - float(self.registers[rs2])

        elif opcode == 'fmul':
            rd, rs1, rs2 = self._parse_r_type(parts[1:])
            self.registers[rd] = float(self.registers[rs1]) * float(self.registers[rs2])

        elif opcode == 'fdiv':
            rd, rs1, rs2 = self._parse_r_type(parts[1:])
            if float(self.registers[rs2]) == 0.0:
                raise ZeroDivisionError("División entre cero en fdiv")
            self.registers[rd] = float(self.registers[rs1]) / float(self.registers[rs2])

        #Intrucciones trigonometricas
        elif opcode == "sin":
            # sin rd, rs1
            rd= self._parse_register(parts[1])
            rs1= self._parse_register(parts[2])
            self.registers[rd] = math.sin(self.registers[rs1])

        elif opcode == "cos":
            # cos rd, rs1
            rd= self._parse_register(parts[1])
            rs1= self._parse_register(parts[2])
            self.registers[rd] = math.cos(self.registers[rs1])

        elif opcode == "tan":
            # tan rd, rs1
            rd= self._parse_register(parts[1])
            rs1= self._parse_register(parts[2])
            self.registers[rd] = math.tan(self.registers[rs1])
            
        # Operaciones lógicas
        elif opcode == 'and':
            rd, rs1, rs2 = self._parse_r_type(parts[1:])
            self.registers[rd] = self.registers[rs1] & self.registers[rs2]
        
        elif opcode == 'or':
            rd, rs1, rs2 = self._parse_r_type(parts[1:])
            self.registers[rd] = self.registers[rs1] | self.registers[rs2]
        
        elif opcode == 'xor':
            rd, rs1, rs2 = self._parse_r_type(parts[1:])
            self.registers[rd] = self.registers[rs1] ^ self.registers[rs2]
        
        # Operaciones de memoria
        elif opcode == 'lw':
            rd, offset, rs1 = self._parse_load(parts[1:])
            address = self.registers[rs1] + offset
            self.registers[rd] = self.memory.get(address, 0)
        
        elif opcode == 'sw':
            rs2, offset, rs1 = self._parse_store(parts[1:])
            address = self.registers[rs1] + offset
            self.memory[address] = self.registers[rs2]
        
        # Saltos condicionales
        elif opcode == 'beq':
            rs1, rs2, label = self._parse_branch(parts[1:])
            if self.registers[rs1] == self.registers[rs2]:
                next_pc = self._resolve_label(label)
        
        elif opcode == 'bne':
            rs1, rs2, label = self._parse_branch(parts[1:])
            if self.registers[rs1] != self.registers[rs2]:
                next_pc = self._resolve_label(label)
        
        elif opcode == 'blt':
            rs1, rs2, label = self._parse_branch(parts[1:])
            if self.registers[rs1] < self.registers[rs2]:
                next_pc = self._resolve_label(label)
        
        elif opcode == 'ble':
            rs1, rs2, label = self._parse_branch(parts[1:])
            if self.registers[rs1] <= self.registers[rs2]:
                next_pc = self._resolve_label(label)
        
        elif opcode == 'bge':
            rs1, rs2, label = self._parse_branch(parts[1:])
            if self.registers[rs1] >= self.registers[rs2]:
                next_pc = self._resolve_label(label)
        
        # Saltos incondicionales
        elif opcode == 'j' or opcode == 'jal':
            label = parts[1]
            next_pc = self._resolve_label(label)
            if opcode == 'jal':
                self.registers[1] = self.pc + 4  # ra = pc + 4
        
        elif opcode == 'jalr':
            rd, offset, rs1 = self._parse_load(parts[1:])
            self.registers[rd] = self.pc + 4
            next_pc = self.registers[rs1] + offset
        
        # Instrucciones especiales
        elif opcode == 'li':
            rd = self._parse_register(parts[1])
            if esEntero(parts[2]):
                imm = int(parts[2])
            elif esFloat(parts[2]):
                imm=float(parts[2])
            self.registers[rd] = imm
        
        elif opcode == 'mv':
            rd = self._parse_register(parts[1])
            rs = self._parse_register(parts[2])
            self.registers[rd] = self.registers[rs]

        elif opcode == "la":
            # Sintaxis esperada: la xN, label
            rd = self._parse_register(parts[1])
            label = parts[2]

            if label not in self.data_labels:
                raise Exception(f"Label '{label}' no definido en la sección .data")

            address = self.data_labels[label]
            self.registers[rd] = address
        
        # Syscalls simplificados
        elif opcode == 'ecall':
            # Si a7 (x17) = 1, imprime el entero en a0 (x10)
            if self.registers[17] == 1:
                print(f"Output: {self.registers[10]}")
            elif self.registers[17] == 2:
                print(f"Output: {self.registers[42]}")
            
            elif self.registers[17] == 4:
                # Mostrar los caracteres almacenados en memoria
                addr = self.registers[10]
                mensaje = []
                while self.memory[addr] != 0:
                    mensaje.append(chr(self.memory[addr]))
                    addr += 1
                print(''.join(mensaje))
            
            elif self.registers[17] == 5:
                try:
                    val = int(input())
                    self.registers[10] = val 
                except ValueError:
                    print("Error: Entrada inválida para entero.")
                    self.registers[10] = 0 
            
            elif self.registers[17] == 6:
                try:
                    val = float(input())
                    self.registers[42] = val 
                except ValueError:
                    print("Error: Entrada inválida para flotante.")
                    self.registers[42] = 0.0 

            elif self.registers[17] == 8:
                addr = self.registers[10]
                max_len = self.registers[11] 
                input_str = input()
                for i, char_code in enumerate(input_str):
                    if i < max_len - 1: 
                        self.memory[addr + i] = ord(char_code)
                    else:
                        break
                self.memory[addr + min(len(input_str), max_len - 1)] = 0 

            # Si a7 (x17) = 10, termina el programa
            elif self.registers[17] == 10:
                self.running = False

            elif self.registers[17]==11:
                if(self.registers[10]==10):
                    addr = self.data_labels["salto"]
                    mensaje = []
                    while self.memory[addr] != 0:
                        mensaje.append(chr(self.memory[addr]))
                        addr += 1
                    print(''.join(mensaje))
                else:
                    print(self.registers[10])

            elif self.registers[17] == 12:
                char_input = input()
                if char_input:
                    self.registers[10] = ord(char_input[0]) 
                else:
                    self.registers[10] = 0

        else:
            if(opcode.startswith(".") or es_id(opcode)):
                print("")
            else:
                print(f"Instrucción no soportada: {instruction}")
        
        # Aseguramos que x0 siempre sea 0
        self.registers[0] = 0
        
        # Actualizamos el PC
        self.pc = next_pc
    
    def _parse_register(self, reg_str):
        """Convierte una cadena de registro a su índice numérico."""
        reg_str = reg_str.strip(',')
        if reg_str == 'zero':
            return 0
        elif reg_str == 'ra':
            return 1
        elif reg_str == 'sp':
            return 2
        elif reg_str == 'gp':
            return 3
        elif reg_str == 'tp':
            return 4
        elif reg_str.startswith('t') and '0' <= reg_str[1] <= '6':
            return 5 + int(reg_str[1])  # t0-t6: 5-11
        elif reg_str == 'fp' or reg_str == 's0':
            return 8
        elif reg_str.startswith('s') and '1' <= reg_str[1] <= '11':
            return 8 + int(reg_str[1])  # s1-s11: 9-19
        elif reg_str.startswith('a') and '0' <= reg_str[1] <= '7':
            return 10 + int(reg_str[1])  # a0-a7: 10-17
        elif reg_str.startswith('x'):
            return int(reg_str[1:])
        elif reg_str.startswith('fa') and '0' <= reg_str[2] <= '7':
            return 42 + int(reg_str[2])  # fa0-fa7: 32-39
        elif reg_str.startswith('f') and reg_str[0:2]!="fa":
            return 32 + int(reg_str[1:])
        else:
            raise ValueError(f"Registro no reconocido: {reg_str}")
    
    def _parse_r_type(self, args):
        """Parsea instrucciones tipo R: add rd, rs1, rs2"""
        rd = self._parse_register(args[0])
        rs1 = self._parse_register(args[1])
        rs2 = self._parse_register(args[2])
        return rd, rs1, rs2
    
    def _parse_i_type(self, args):
        """Parsea instrucciones tipo I: addi rd, rs1, imm"""
        rd = self._parse_register(args[0])
        rs1 = self._parse_register(args[1])
        imm = int(args[2])
        return rd, rs1, imm
    
    def _parse_load(self, args):
        """Parsea instrucciones de carga: lw rd, offset(rs1)"""
        rd = self._parse_register(args[0])
        offset_base = args[1].split('(')
        offset = int(offset_base[0])
        rs1 = self._parse_register(offset_base[1].strip(')'))
        return rd, offset, rs1
    
    def _parse_store(self, args):
        """Parsea instrucciones de almacenamiento: sw rs2, offset(rs1)"""
        rs2 = self._parse_register(args[0])
        offset_base = args[1].split('(')
        offset = int(offset_base[0])
        rs1 = self._parse_register(offset_base[1].strip(')'))
        return rs2, offset, rs1
    
    def _parse_branch(self, args):
        """Parsea instrucciones de salto condicional: beq rs1, rs2, label"""
        rs1 = self._parse_register(args[0])
        rs2 = self._parse_register(args[1])
        label = args[2]
        return rs1, rs2, label
    
    def _resolve_label(self, label):
        """Resuelve una etiqueta a su dirección correspondiente."""
        if label in self.memory:
            return self.memory[label]
        try:
            # Intenta interpretar como una dirección absoluta
            return int(label)
        except ValueError:
            raise ValueError(f"Etiqueta no encontrada: {label}")
    
    def print_state(self):
        """Imprime el estado actual de los registros."""
        print("Estado de registros:")
        for i in range(64):
            alias = ""
            if i == 0:
                alias = "zero"
            elif i == 1:
                alias = "ra"
            elif i == 2:
                alias = "sp"
            elif i == 10:
                alias = "a0"
            elif i == 20:
                alias = "a7"
            elif i == 42:
                alias = "fa0"
            elif i == 49:
                alias = "fa7"
            if(i<32):
                if alias:
                    print(f"x{i} ({alias}): {self.registers[i]}")
                else:
                    print(f"x{i}: {self.registers[i]}")
            if(i>=32):
                if alias:
                    print(f"f{i-32} ({alias}): {self.registers[i]}")
                else:
                    print(f"f{i-32}: {self.registers[i]}")

# Compilador C

class Variable:
    def __init__(self, nombre, tipo):
        self.nombre = nombre
        self.tipo = tipo
        self.valor = None

def agrega_var(tabla_var, nombre, tipo):
    tabla_var.append(Variable(nombre, tipo))
    pass

def existe_var(tabla_var, nombre):
    encontrado = False
    for v in tabla_var:
        if v.nombre == nombre:
            encontrado = True
    return encontrado

def set_var(tabla_var, nombre, valor):
    if existe_var(tabla_var, nombre):
        for v in tabla_var:
            if v.nombre == nombre:
                v.valor = valor
    else:
        print('variable ', nombre, 'no encontrada')
        return None

def imprime_tabla_var(tabla_var):
    print()
    print('   Tabla de variables')
    print('nombre\t\ttipo\t\tregistro')
    for v in tabla_var:
        print(v.nombre,'\t\t', v.tipo,'\t\t', v.valor)
    return None
    
def getValor(tabla_var, varNombre):
    for v in tabla_var:
        if (v.nombre == varNombre):
            return v.valor
    return None

def getTipo(tabla_var, varNombre):
    for v in tabla_var:
        if (v.nombre == varNombre):
            return v.tipo
    return None

def es_simbolo_esp(caracter):
    return caracter in "+-*;,.:!#=%&/(){}[]<><=>=="

def es_separador(caracter):
    return caracter in " \n\t"

def es_id(cad):
    return (cad[0] in "_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

def esEntero(cad):  
    return cad.isdigit()

def esFloat(cad):
    try:
        float(cad)
        return '.' in cad or 'e' in cad.lower()
    except ValueError:
        return False

def  es_pal_res(cad):
    palres = ["var", "int", "real", "string", "char" , "float", 'print', 'println', 'read', 'tabla', 'main', 'void', 'if', 'else', 'in', 'end', 'for', 'sin', 'cos', 'tan']
    return (cad in palres)

def  es_tipo(cad):
    tipos = ["int", "real", "string", "char", "float"]
    return (cad in tipos)

def tipoResultado(op1_tipo, operador, op2_tipo):
    '''
    Devuelve el tipo del resultado según el tipo de los operandos y el operador.
    Tipos válidos: 'int', 'float'
    Operadores válidos: '+', '-', '*', '/'
    '''
    if op1_tipo == 'int' and op2_tipo == 'int':
        if operador == '/':
            return 'float'
        else:
            return 'int'
    else:
        # Cualquier combinación donde al menos uno sea float da float
        return 'float'

def obtenerPrioridadOperador(e):
    prioridades = {
            '+': 1,
            '-': 1,
            '*': 2,
            '/': 2,
            '^': 3
        }
    return prioridades.get(e, 0)


def convertirInfijaAPostfija(infija):
	'''Convierte una expresión infija a una posfija, devolviendo una lista.'''
	pila = []
	salida = []
	for e in infija:
		if e == '(':
			pila.append(e)
		elif e == ')':
			while pila[len(pila) - 1 ] != '(':
				salida.append(pila.pop())
			pila.pop()
		elif e in ['+', '-', '*', '/', '^']:
			while (len(pila) != 0) and obtenerPrioridadOperador(e) <= obtenerPrioridadOperador(pila[len(pila) - 1]):
				salida.append(pila.pop())
			pila.append(e)
		else:
			salida.append(e)
	while len(pila) != 0:
		salida.append(pila.pop())
	return salida

def analizarPostfija(postfija):
    '''Recibe una lista en notación postfija y devuelve una lista con los pasos de operaciones.'''
    pila = []
    pasos = []
    temp_var = 1  # Para nombrar resultados temporales si quieres

    for token in postfija:
        t=[]
        if token not in ['+', '-', '*', '/', '^']:
            pila.append(token)
        else:
            # Se sacan los dos últimos operandos
            op2 = pila.pop()
            op1 = pila.pop()
            resultado = f"t{temp_var}"
            t=[resultado,"=",op1,token,op2]
            pasos.append(t)
            pila.append(resultado)
            temp_var += 1
    print(postfija)
    print(pasos)
    return pasos

def procesar_for(tokens_for, cuerpo, tabla_var, f, x):
    tokens_for = [t for t in tokens_for if t not in ['(', ')', ';']]
    program = ""

    var_control = tokens_for[1]
    valor_inicial = tokens_for[3]
    condicion = tokens_for[4:7]  # ['i', '<', '5']
    incremento = tokens_for[7:]  # Esto no se usa directamente aún

    # === Inicialización ===
    if not existe_var(tabla_var, var_control):
        reg_control = f"x{x}"
        program += f"li {reg_control}, {valor_inicial}\n"
        agrega_var(tabla_var, var_control, "int")
        set_var(tabla_var, var_control, reg_control)
        x += 1
    else:
        reg_control = getValor(tabla_var, var_control)
        program += f"li {reg_control}, {valor_inicial}\n"

    # === Límite (condición) ===
    op_derecho = condicion[2]
    if op_derecho == var_control:
        raise ValueError("La condición del for no puede ser contra sí misma")

    if existe_var(tabla_var, op_derecho):
        reg_limite = getValor(tabla_var, op_derecho)
    elif esEntero(op_derecho) or esFloat(op_derecho):
        reg_limite = f"x{x}"
        program += f"li {reg_limite}, {op_derecho}\n"
        x += 1
    else:
        raise ValueError(f"Operando no válido en condición del for: {op_derecho}")

    # === Etiquetas únicas ===
    unique_id = f
    inicio = f"for_start_{var_control}_{unique_id}"
    cuerpo_label = f"for_body_{var_control}_{unique_id}"
    fin = f"for_end_{var_control}_{unique_id}"

    # === Código del for ===
    program += f"{inicio}:\n"
    program += f"blt {reg_control}, {reg_limite}, {cuerpo_label}\n"
    program += f"j {fin}\n"
    program += f"{cuerpo_label}:\n"
    program += cuerpo
    program += f"addi {reg_control}, {reg_control}, 1\n"
    program += f"j {inicio}\n"
    program += f"{fin}:\n"

    return program, x, f


def get_operand(tabla_var, expr, reg_prefix):
    expr = expr.strip().strip(';')  # Elimina espacios y ;
    if existe_var(tabla_var, expr):
        return getValor(tabla_var, expr)
    elif esFloat(expr):
        return expr
    elif esEntero(expr):
        return expr
    else:
        raise ValueError(f"Operando no reconocido o no soportado: {expr}")


def quitar_comentarios(program):
    estado = 'Z'
    texto = ''
    for letra in program.strip():
        if estado == 'Z':
            if letra == '/':
                estado = 'A'
            else:
                texto += letra
        elif estado == 'A':
            if letra == '*':
                estado = 'B'
            else:
                estado = 'Z'
                texto += '/'
        elif estado == 'B':
            if letra == '*':
                estado = 'C'
        elif estado == 'C':
            if letra == '/':
                estado = 'Z'
        elif letra != '*':
            estado = 'B'
    return texto

def separa_tokens(program):
    tokensP=[]
    for line in program.strip().split('\n'):
        line=line.strip()
        if line=="" or line=="}":
            line=";"
        if len(program) < 3:
            return []
        elif line[-1] != ';' and not (line.startswith("for")):
            print('Error, la siguiente linea no tiene ";": '+line)
            break
        else:    
            tokens = []
            tokens2 = []    
            dentro = False   
            for l in line:
                if es_simbolo_esp(l) and not(dentro):
                    tokens.append(l)
                if (es_simbolo_esp(l) or es_separador(l)) and dentro:
                    tokens.append(cad)
                    dentro = False
                    if es_simbolo_esp(l):
                        tokens.append(l)
                if not (es_simbolo_esp(l)) and not (es_separador(l)) and not(dentro):
                    dentro = True
                    cad=""
                if not (es_simbolo_esp(l)) and not (es_separador(l)) and dentro:
                        cad = cad + l   
            compuesto = False
            for c in range(len(tokens)-1):
                if compuesto:
                    compuesto = False
                    continue
                if tokens[c] in "=<>!" and tokens[c+1]=="=":
                    tokens2.append(tokens[c]+"=")
                    compuesto = True
                else:
                    tokens2.append(tokens[c])
            if tokens:
                tokens2.append(tokens[-1])   
            for c in range(1,len(tokens2)-1):
                if tokens2[c]=="." and esEntero(tokens2[c-1]) and esEntero(tokens2[c+1]):
                    tokens2[c]=tokens2[c-1]+tokens2[c]+tokens2[c+1]
                    tokens2[c-1]="borrar"
                    tokens2[c+1]="borrar"    
            porBorrar = tokens2.count("borrar")
            for c in range(porBorrar):
                tokens2.remove("borrar")
            tokens=[]
            dentroCad = False
            cadena = ""
            for t in tokens2:        
                if dentroCad:
                    if t[-1]=='"':
                        cadena=cadena+" "+t
                        tokens.append(cadena[1:-1])
                        dentroCad = False
                    else:
                        cadena = cadena+" "+t
                elif ((t[0]=='"')):
                    cadena= t
                    dentroCad = True
                else:
                    tokens.append(t)
            tokensP.append(tokens)
    return tokensP


def correr_programa(tabla_var,tokens,simulator):
    seccion_data={}
    program=""
    x=1
    f=0
    i=0
    while i <= len(tokens)-1:
        contenido=[]
        if es_id(tokens[i][0]):
            if es_pal_res(tokens[i][0]):
                if es_tipo(tokens[i][1]):
                    if tokens[i][1]=="float":
                        if es_id(tokens[i][2]):
                            if existe_var(tabla_var, tokens[i][2]):
                                print('Error en la linea numero '+ str(i+1)+', esta variable ya fue declarada: ' + tokens[i][2] )
                                break
                            else:
                                agrega_var(tabla_var, tokens[i][2], tokens[i][1])
                                program+="li f"+str(f)+", 0.0\n"
                                set_var(tabla_var, tokens[i][2],"f"+str(f))
                                f+=1
                    elif es_id(tokens[i][2]):
                        if(existe_var(tabla_var, tokens[i][2])):
                            print('Error en la linea numero '+ str(i+1)+', esta variable ya fue declarada: ' + tokens[i][2] )
                            break
                        else:
                            agrega_var(tabla_var, tokens[i][2], tokens[i][1])
                            program+="li x"+str(x)+", 0\n"
                            set_var(tabla_var, tokens[i][2],"x"+str(x))
                            x+=1
                elif tokens[i][0] == 'read':
                    if tokens[i][1] == '(' and es_id(tokens[i][2]) and tokens[i][3] == ')':
                        var_name = tokens[i][2]
                        var_type = getTipo(tabla_var, var_name)
                        var_reg = getValor(tabla_var, var_name)

                        if not var_type or not var_reg:
                            print(f"Error: variable '{var_name}' no encontrada para lectura.")
                            continue

                        if var_type == 'int':
                            program += "li a7, 5\n" 
                            program += "ecall\n"
                            program += f"mv {var_reg}, a0\n" 
                        elif var_type == 'float':
                            program += "li a7, 6\n" 
                            program += "ecall\n"
                            program += f"mv {var_reg}, fa0\n" 
                        elif var_type == 'char':
                            program += "li a7, 12\n" 
                            program += "ecall\n"
                            program += f"mv {var_reg}, a0\n"
                        elif var_type == 'string':
                            buffer_label = var_reg 
                            max_string_len = 256
                            program += f"la a0, {buffer_label}\n" 
                            program += f"li a1, {max_string_len}\n"
                            program += "li a7, 8\n" 
                            program += "ecall\n"
                        else:
                            print(f"Error: Tipo de variable '{var_type}' no soportado para lectura.")   

                elif tokens[i][0] == 'tabla':               
                    imprime_tabla_var(tabla_var)
                    
                # Manejo de print y println
                elif tokens[i][0] in ['print', 'println']:
                    if len(tokens[i]) < 3 or tokens[i][1] != '(' or tokens[i][-2] != ')':
                        print("Error de sintaxis en print/println")
                    
                    for c in range(2,len(tokens[i])-1):
                        if(tokens[i][c]!=',' and tokens[i][c]!=')'):
                            contenido.append(tokens[i][c])
                        c+=1
                    for j in range(len(contenido)):
                        imprimir=contenido[j]
                        if (tokens[i][-2]==")" and existe_var(tabla_var,imprimir)==False): #Hay un solo token dentro del print
                            
                            # Cadena literal
                            cadena = imprimir
                            label = f"str_{len(seccion_data)}"
                            seccion_data[label]=cadena
                            simulator.load_data_label(label, cadena)
                            simulator.execute_la(10, label)
                            program+=(f"la a0, {label}\n")# Cargar dirección de la cadena
                            program+=("li a7, 4\n") # Syscall para Write('texto')"
                            program+=("ecall\n") # Ejecutar syscall

                        
                        elif tokens[i][-2]==")" and existe_var(tabla_var,imprimir)==True:
                            # Variable
                            tipo = getTipo(tabla_var, imprimir)
                            registro = getValor(tabla_var, imprimir)
                            
                            if not tipo or not registro:
                                print(f"Error: variable '{imprimir}' no encontrada")
                                continue
                            
                            if tipo == "int":
                                program+=(f"mv a0, {registro}\n") # Cargar valor de {tokens[i][2]}"
                                program+=("li a7, 1\n") # Syscall para Write(integer)
                                program+=("ecall\n") # Ejecutar syscall
                            
                            elif tipo == "float":
                                program+=(f"mv fa0, {registro}\n")  # Cargar valor de {tokens[i][2]}
                                program+=("li a7, 2\n")  # Cargar valor de {tokens[i][2]}
                                program+=("ecall\n") # Ejecutar syscall
                            
                            elif tipo == "char":
                                program+=(f"mv a0, {registro}\n")# Cargar valor de {tokens[i][2]}
                                program+=("li a7, 11\n") # Syscall para Write(carácter)
                                program+=("ecall\n") # Ejecutar syscall
                            
                            elif tipo == "string":
                                program+=(f"la a0, buffer_{imprimir}\n") # Cargar dirección de {tokens[i][2]}
                                program+=("li a7, 4\n") # Syscall para Write('texto')
                                program+=("ecall\n") # Syscall para Write('texto')
                        j+=1

                    if(tokens[i][0] == 'println'):
                        seccion_data["salto"]="\n"
                        simulator.load_data_label("salto","\n")
                        simulator.execute_la(10,'salto')
                        program+="li a0, 10\n"
                        program+="li a7, 11\n"
                        program+="ecall\n"

                elif tokens[i][0] == 'for':                    
                    linea = tokens[i][:-2]  # Línea donde inicia el for
                    cuerpo_for = ""    # Código que irá dentro del for
                    i += 1  # Avanza a la siguiente línea después del 'for'

                    # Recorrer todas las líneas hasta encontrar 'endfor'
                    while i < len(tokens) and tokens[i][0] != '}':
                        print("CUERPO FOR:", tokens[i])  # Debug

                        # Detectar multiplicación del tipo: ID = ID * valor
                        if len(tokens[i]) >= 5 and tokens[i][3] == '*':
                            reg1 = getValor(tabla_var, tokens[i][2])
                            reg2 = get_operand(tabla_var, tokens[i][4], "x")
                            if esEntero(tokens[i][4]):
                                temp = f"x{x}"
                                cuerpo_for += f"li {temp}, {tokens[i][4]}\n"
                                x += 1
                                reg2 = temp
                            reg = getValor(tabla_var, tokens[i][0])
                            cuerpo_for += f"mul {reg}, {reg1}, {reg2}\n"

                        # Print de variables o cadenas
                        elif tokens[i][0] == 'print' and len(tokens[i]) >= 3:
                            var = tokens[i][2]
                            if existe_var(tabla_var, var):
                                tipo = getTipo(tabla_var, var)
                                registro = getValor(tabla_var, var)
                                if tipo == "int":
                                    cuerpo_for += f"mv a0, {registro}\nli a7, 1\necall\n"
                                elif tipo == "float":
                                    cuerpo_for += f"mv fa0, {registro}\nli a7, 2\necall\n"
                                elif tipo == "char":
                                    cuerpo_for += f"mv a0, {registro}\nli a7, 11\necall\n"
                                elif tipo == "string":
                                    cuerpo_for += f"la a0, {registro}\nli a7, 4\necall\n"
                            else:
                                # Si no es variable, asumir que es cadena literal
                                msg = tokens[i][2]
                                label = f"msg_{i}"
                                simulator.load_data_label(label, msg)
                                cuerpo_for += f"la a0, {label}\nli a7, 4\necall\n"

                        i += 1  # Avanza a la siguiente instrucción dentro del for

                    # Al salir del while, estamos en 'endfor'
                    ensamblador_for, x, f = procesar_for(linea, cuerpo_for, tabla_var, f, x)
                    program += ensamblador_for
                    i += 1  # Saltamos el 'endfor' para continuar

                    continue

                elif(tokens[i][0]=='end'):
                    program+="li a7, 10\n"
                    program+="ecall\n"

                elif(tokens[i][0] in ['sin','cos','tan']):
                    if(tokens[i][0]=='sin'):
                        program+="sin fa0, "+str(getValor(tabla_var,tokens[i][2])+"\n")
                        program+=("li a7, 2\n")  # Cargar valor de {tokens[i][2]}
                        program+=("ecall\n") # Ejecutar syscall
                    elif(tokens[i][0]=='cos'):
                        program+="cos fa0, "+str(getValor(tabla_var,tokens[i][2])+"\n")
                        program+=("li a7, 2\n")  # Cargar valor de {tokens[i][2]}
                        program+=("ecall\n") # Ejecutar syscall
                    elif(tokens[i][0]=='tan'):
                        program+="tan fa0, "+str(getValor(tabla_var,tokens[i][2])+"\n")
                        program+=("li a7, 2\n")  # Cargar valor de {tokens[i][2]}
                        program+=("ecall\n") # Ejecutar syscall
            
            elif (len(tokens[i])==4): # Es de la forma "ID = valor;"
                if (tokens[i][1]=="="): #Se verifica si tiene el caracter "="
                    program+=f"li {getValor(tabla_var,tokens[i][0])}, {tokens[i][2]}\n"
            elif(len(tokens[i])>=5):
                print(tokens[i])
                if (tokens[i][1]=="="): #Se verifica si tiene el caracter "="
                    valores=[]
                    e1=tokens[i][2:-1]
                    valores=analizarPostfija(convertirInfijaAPostfija(e1))
                    for j in range(len(valores)):
                        if (j<len(valores)-1):
                            agrega_var(tabla_var, valores[j][0], tipoResultado(valores[j][2],valores[j][3],valores[j][4])) #Se asigna el nuevo valor
                        if(tipoResultado(valores[j][2],valores[j][3],valores[j][4])=="float"):
                            program+="li f"+str(f)+", 0.0\n"
                            set_var(tabla_var, valores[j][0] ,"f"+str(f))
                            f+=1
                        else:
                            program+="li x"+str(x)+", 0\n"
                            set_var(tabla_var,valores[j][0] ,"x"+str(x))
                            x+=1
                        if(valores[j][3] in '+-*/'):
                            registro=""
                            if (tipoResultado(valores[j][2],valores[j][3],valores[j][4])=="float"):
                                registro="f"+str(f-1)
                            else:
                                registro="x"+str(x-1)
                            if(j==len(valores)-1):
                                registro=str(getValor(tabla_var, tokens[i][0]))
                            if(valores[j][3]=='-'):
                                program+="sub "+str(registro)+", "+str(getValor(tabla_var,valores[j][2]))+", "+str(getValor(tabla_var,valores[j][4])+"\n")
                            elif(valores[j][3]=='+'):
                                program+="add "+str(registro)+", "+str(getValor(tabla_var,valores[j][2]))+", "+str(getValor(tabla_var,valores[j][4])+"\n")
                            elif(valores[j][3]=='*'):
                                if(getValor(tabla_var,valores[j][4])!=None):
                                    program+="mul "+str(registro)+", "+str(getValor(tabla_var,valores[j][2]))+", "+str(getValor(tabla_var,valores[j][4])+"\n")
                                else:
                                    program+="muli "+str(registro)+", "+str(getValor(tabla_var,valores[j][2]))+", "+valores[j][4]+"\n"
                            elif(valores[j][3]=='/'):
                                program+="div "+str(registro)+", "+str(getValor(tabla_var,valores[j][2]))+", "+str(getValor(tabla_var,valores[j][4])+"\n")
                    if(tokens[i][2] in ['sin','cos','tan']):
                        if(tokens[i][2]=='sin'):
                            program+="sin fa0, "+str(getValor(tabla_var,tokens[i][4])+"\n")
                            program+=("mv "+str(getValor(tabla_var,tokens[i][0]))+", fa0\n")
                        elif(tokens[i][2]=='cos'):
                            program+="cos fa0, "+str(getValor(tabla_var,tokens[i][4])+"\n")
                            program+=("mv "+str(getValor(tabla_var,tokens[i][0]))+", fa0\n")
                        elif(tokens[i][2]=='tan'):
                            program+="tan fa0, "+str(getValor(tabla_var,tokens[i][4])+"\n")
                            program+=("mv "+str(getValor(tabla_var,tokens[i][0]))+", fa0\n")
        i += 1                  
    
    codigo_final = ".data\n"
    for linea in seccion_data:
        codigo_final += linea + "\n"
    codigo_final += "\n.text\n" + program
    print("-------------CODIGO ENSAMBLADOR---------------")
    print(codigo_final)
    simulator.load_program(codigo_final)
    print("-------------TABLA FINAL DE VARIABLES---------------")
    imprime_tabla_var(tabla_var)
    simulator.run()

# Función para ejecutar un programa
def compilarC(program_code):
    programa=quitar_comentarios(program_code)
    #print(programa)
    tokens=separa_tokens(programa)
    #print(tokens)
    tabla_var = []
    simulator=RiscVSimulator()
    correr_programa(tabla_var,tokens,simulator)

# Función principal
if __name__ == "__main__":
    # Este es solo un ejemplo - los programas reales se cargarán desde archivos o entrada del usuario
    rad=(math.pi/3)
    texto = """
var float rad;
print("Hola Mundo");
var float x1;
var float x2; /* coordenadas del 1er punto */
var float y1;
var float y2; /* coordenadas del 2do punto */
var float m; /* pendiente de la recta */
print("Escriba el valor de x1: ");
read(x1);
print("Escriba el valor de x2: ");
read(x2);
print("Escriba el valor de y1: ");
read(y1);
print("Escriba el valor de y2: ");
read(y2);
m = (y2 - y1) / (x2 - x1);
println("La pendiente es: ", m);
"""
    texto+=f"rad={rad};"
    texto+="""
sin(rad);
cos(rad);
tan(rad);
var int i;
var int x;
for(i = 0; i < 5; i = i + 1){
    x = i * 2;
    print("El valor de x es:");
    print(x);
};
end;
"""
    compilarC(texto)