class RiscVSimulator:
    """Simulador simplificado de RISC-V que soporta un subconjunto básico de instrucciones."""
    
    def __init__(self):
        # Registros: x0-x31 (x0 siempre es 0)
        self.registers = [0] * 32
        # Memoria (simulada como un diccionario para acceso eficiente)
        self.memory = {}
        # Contador de programa
        self.pc = 0
        # Diccionario de instrucciones
        self.instructions = {}
        # Flag para controlar ejecución
        self.running = False
    
    def load_program(self, program):
        """Carga un programa en la memoria del simulador."""
        # Limpiamos el estado previo
        self.registers = [0] * 32
        self.memory = {}
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
        print(program)
    
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
        
        # Syscalls simplificados
        elif opcode == 'ecall':
            # Si a7 (x17) = 1, imprime el entero en a0 (x10)
            if self.registers[17] == 1:
                print(f"Output: {self.registers[10]}")
            # Si a7 (x17) = 10, termina el programa
            elif self.registers[17] == 10:
                self.running = False
        
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
        elif reg_str.startswith('f'):
            return int(reg_str[1:])
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
        for i in range(32):
            alias = ""
            if i == 0:
                alias = "zero"
            elif i == 1:
                alias = "ra"
            elif i == 2:
                alias = "sp"
            elif i == 10:
                alias = "a0"
            elif i == 17:
                alias = "a7"
            
            if alias:
                print(f"x{i} ({alias}): {self.registers[i]}")
            else:
                print(f"x{i}: {self.registers[i]}")

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
    palres = ["var", "int", "real", "string", "char" , "float", 'print', 'println', 'read', 'tabla', 'main', 'void', 'if', 'else', 'in']
    return (cad in palres)

def  es_tipo(cad):
    tipos = ["int", "real", "string", "char", "float"]
    return (cad in tipos)

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
        if len(program) < 3:
            return []
        elif line[-1] != ';':
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
    program=""
    x=0
    f=0
    for i in range(len(tokens)-1):
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
                        leido = input()
                        if getTipo(tabla_var,tokens[i][2])=="float" and leido.isdigit():
                            leido+='.0'
                            float(leido)
                        program+="li "+getValor(tabla_var,tokens[i][2])+", "+leido+"\n"    
                elif tokens[i][0] == 'tabla':               
                    imprime_tabla_var(tabla_var)
                elif (tokens[i][0]=="print" or tokens[i][0]=="println"):# El renglon es un print
                    if (tokens[i][3]==")"): #Hay un solo token dentro del print
                        if(existe_var(tabla_var,tokens[i][2])):
                            valor = getValor(tabla_var, tokens[i][2]); #recuperamos el valor de la variable
                            print(valor)
                            if(tokens[i][0]=="println"):
                                print('\n')
                        else:
                            print(tokens[i][2])
                            if(tokens[i][0]=="println"):
                                print('\n')
            elif (len(tokens[i])==4): # Es de la forma "ID = valor;"
                if (tokens[i][1]=="="): #Se verifica si tiene el caracter "="
                    set_var(tabla_var, tokens[i][0], tokens[i][2]) #Se asigna el nuevo valor
    simulator.load_program(program)
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
    texto = """
var float x1;
var float x2; /* coordenadas del 1er punto */
var float y1;
var float y2; /* coordenadas del 2do punto */
var float m; /* pendiente de la recta */
tabla;
print("Escriba el valor de x1: ");
read(x1);
print("Escriba el valor de x2: ");
read(x2);
print("Escriba el valor de y1: ");
read(y1);
print("Escriba el valor de y2: ");
read(y2);
tabla;
m = (y2 - y1) / (x2 - x1);
println("La pendiente es: ", m);
end;
"""
    compilarC(texto)