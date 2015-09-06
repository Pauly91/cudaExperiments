
import math as m
for i in range(0,8):
	for j in range(0,8):
		if j == 0:
			print str(m.cos((j*m.pi*(2*i + 1))/16)/m.sqrt(8)) + ',', 
		else:
			print str( m.sqrt(2) * m.cos((j*m.pi*(2*i + 1))/16)/m.sqrt(8)) + ',',
	print "\n"