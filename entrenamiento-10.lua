require 'rnn'
require 'math'
require 'funciones'

math.randomseed(os.time())

epocas = 1000
lr = 0.000001
largo_maximo = 10

print('Abriendo archivos')

CARPETAS = {'../Datos procesados/semana/','../Datos procesados/viernes/'}
MESES = {'03-MARZO','04-ABRIL','05-MAYO','06-JUNIO','07-JULIO','08-AGOSTO','09-SEPTIEMBRE','10-OCTUBRE','11-NOVIEMBRE','12-DICIEMBRE'}
FERIADOS = {{1,1},{2,2},{4,3},{4,21},{5,1},{9,7},{10,12},{11,2},{12,25}}
archivo_validacion = io.open('./'..arg[1], 'r')


if archivo_validacion == nil then
	print('No se ha encontrado el archivo de validacion especificado! ./'..arg[1])
	os.exit()
end

viajes = {}
for c = 1, #CARPETAS do
	for m = 1, #MESES do
		viaje = {}
		
		print(CARPETAS[c]..MESES[m]..'-I.txt')
		archivo = io.open(CARPETAS[c]..MESES[m]..'-I.txt','r')
		
		linea = archivo:read()

		while linea ~= nil do
			linea = split(linea, '\t')
			if linea[1] ~= 'F' then
				dd = tonumber(linea[1])
				 t = tonumber(linea[4])
				 d = tonumber(linea[3])
				table.insert(viaje, {dd, t, d})
			else
				if #viaje <= 91 then --si la duracion del viaje es de hasta 90 minutos
					if not feriado(FERIADOS,tonumber(linea[3]),tonumber(linea[2])) then
						T = tonumber(linea[5])*3600 + tonumber(linea[6])*60 + tonumber(linea[7])
						D = tonumber(linea[9])
						F = {tonumber(linea[2]),tonumber(linea[3]),tonumber(linea[5]),tonumber(linea[6]),tonumber(linea[7])} --dia,mes,hora,minuto,segundo
						table.insert(viaje, {T,D})
						table.insert(viaje,F)
						table.insert(viajes, viaje)
					end
				end
				viaje = {}
			end
			linea = archivo:read()
		end
		archivo:close()
	end
end

print('Procesando datos')

viajesX = {}
viajesY = {}
datos = {}

for i = 1,#viajes do
	X = {}
	Y = {}
	T = viajes[i][#viajes[i]-1][1]
	D = viajes[i][#viajes[i]-1][2]
	for j = 1, #viajes[i] - 12 do
		dd = viajes[i][j][1]
		 t = viajes[i][j][2]
		 d = viajes[i][j][3]
		 y = viajes[i][j+10][3] --la funcion objetivo intenta obtener el porcentaje de avance del bus en el minuto t+1
		--normalizacion
		table.insert(X, torch.DoubleTensor({T/86400, t/86400, dd/D, d/D})) --t0, t, dd, d
		table.insert(Y, torch.DoubleTensor({100*y/D})) --d(t+1)
	end
	table.insert(datos, viajes[i][#viajes[i]])
	table.insert(viajesX, X)
	table.insert(viajesY, Y)
end

viajes = nil
collectgarbage()

--seleccion del conjunto de validacion
linea = archivo_validacion:read()
validacionX = {}
validacionY = {}

while linea ~= nil do
	linea = split(linea, '\t')
	linea[1] = tonumber(linea[1])
	linea[2] = tonumber(linea[2])
	linea[3] = tonumber(linea[3])
	linea[4] = tonumber(linea[4])
	linea[5] = tonumber(linea[5])
	for i = 1, #datos do
		if datos[i][1] == linea[1] and datos[i][2] == linea[2] and datos[i][3] == linea[3] and datos[i][4] == linea[4] and datos[i][5] == linea[5] then
			table.insert(validacionX, viajesX[i])
			table.insert(validacionY, viajesY[i])
			table.remove(viajesX,i)
			table.remove(viajesY,i)
			table.remove(datos,i)
			break
		end
	end
	linea = archivo_validacion:read()
end

--generacion de la red neuronal

print('Generando red')

red = nn.Sequential()
red:add(nn.Sequencer(nn.LSTM(4,100)))
red:add(nn.Sequencer(nn.LSTM(100,100)))
red:add(nn.Sequencer(nn.Linear(100,100)))
red:add(nn.Sequencer(nn.Linear(100,100)))
red:add(nn.Sequencer(nn.Linear(100,1)))

criterio = nn.SequencerCriterion(nn.MSECriterion())

tester = nn.MSECriterion()

print(red)
print(criterio)
print(tester)

--Separacion de los folds

foldsX = {{},{},{},{},{},{},{},{},{},{}}
foldsY = {{},{},{},{},{},{},{},{},{},{}}

while #viajesX > 0 do
	for i = 1, 10 do
		if #viajesX == 0 then
			break
		end
		indice = math.random(#viajesX)
		table.insert(foldsX[i], viajesX[indice])
		table.insert(foldsY[i], viajesY[indice])
		table.remove(viajesX, indice)
		table.remove(viajesY, indice)
	end
end

viajesX = nil
viajesY = nil
collectgarbage()



err_fold = 0
err_epoca = 0
--validacion inicial

acumulador = 0
puntos = 0
for v = 1, #validacionX do
	if v % 50 == 0 then
		os.execute('clear')
		print('validacion inicial:', math.floor(100*v/#validacionX)..'%')
	end
	viajeX = validacionX[v]
	viajeY = validacionY[v]
	largo_viaje = #viajeX
	contador = 0
	while contador < largo_viaje do
		largo_sec = 0
		if largo_viaje - contador >= largo_maximo then
			largo_sec = math.random(largo_maximo)
		else
			largo_sec = math.random(largo_viaje - contador)
		end
		y = {}
		x = {}
		for i = 1, largo_sec do
			contador = contador + 1
			table.insert(x, viajeX[contador])
			table.insert(y, viajeY[contador])
		end
		salida = red:forward(x)
		red:forget()
		salida = salida[#salida] --el ultimo tensor de la salida
		objetivo = y[#y]--el ultimo tensor en el vector objetivo
		err = tester:forward(salida, objetivo)
		acumulador = acumulador + err
		puntos = puntos + 1
	end
end
err_epoca = acumulador / puntos
err_fold = acumulador / puntos

registro = io.open('./registro/entrenamiento-10.txt','a')
registro:write('0\t0\t'..err_fold..'\n')
registro:close()

--entrenamiento
for e = 1, epocas do
	for f = 1, 10 do
		for v = 1, #foldsX[f] do
			if v % 10 == 0 then
				os.execute('clear')
				print('Epoca:',e)
				print('Fold:',f,'Entrenando',math.floor(100*v/#foldsX[f])..'%')
				print('Error Epoca anterior:', err_epoca)
				print('Error Fold anterior:',err_fold)
				
			end
			viajeX = foldsX[f][v]
			viajeY = foldsY[f][v]
			--generacion de las secuencias de largo aleatorio
			largo_viaje = #viajeX
			contador = 0
			while contador < largo_viaje do
				largo_sec = 0
				if largo_viaje - contador >= largo_maximo then
					largo_sec = math.random(largo_maximo)
				else
					largo_sec = math.random(largo_viaje - contador)
				end
				x = {}
				y = {}
				for i = 1, largo_sec do
					contador = contador + 1
					table.insert(x, viajeX[contador])
					table.insert(y, viajeY[contador])
				end
				gradientUpgrade(red,x,y,criterio,lr)
				red:forget()
			end			
		end
		
		--validacion luego del fold
		acumulador = 0
		puntos = 0
		for v = 1, #validacionX do
			if v % 50 == 0 then
				os.execute('clear')
				print('Epoca:',e)
				print('Fold:',f)
				print('Error Epoca anterior:', err_epoca)
				print('Error Fold anterior:',err_fold)
				print('Validando Entrenamiento',math.floor(100*v/#validacionX[v])..'%')
			end
			viajeX = validacionX[v]
			viajeY = validacionY[v]
			largo_viaje = #viajeX
			contador = 0
			while contador < largo_viaje do
				largo_sec = 0
				if largo_viaje - contador >= largo_maximo then
					largo_sec = math.random(largo_maximo)
				else
					largo_sec = math.random(largo_viaje - contador)
				end
				y = {}
				x = {}
				for i = 1, largo_sec do
					contador = contador + 1
					table.insert(x, viajeX[contador])
					table.insert(y, viajeY[contador])
				end
				salida = red:forward(x)
				red:forget()
				salida = salida[#salida] --el ultimo tensor de la salida
				objetivo = y[#y]--el ultimo tensor en el vector objetivo
				err = tester:forward(salida, objetivo)
				acumulador = acumulador + err
				puntos = puntos + 1
			end
		end
		err_fold = acumulador / puntos
		registro = io.open('./registro/entrenamiento-10.txt','a')
		registro:write(e..'\t'..f..'\t'..err_fold..'\n')
		registro:close()
		torch.save('./redes/prediccion-10.rn',red)
	end
	err_epoca = err_fold
end


