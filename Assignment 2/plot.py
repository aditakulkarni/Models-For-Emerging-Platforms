import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

objects = ('2^15 <U,V>','2^15 <2*U,V>','2^15 <U,2*V>','2^15 <2*U,2*V>','2^15 <U/2,V>','2^15 <U,V/2>','2^15 <U/2,V/2>')
y_pos = np.arange(len(objects))
performance = [0.014,0.014,0.014,0.015,0.014,0.014,0.014]
 
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Time in milliseconds')
plt.title('Execution times')
 
plt.show()
