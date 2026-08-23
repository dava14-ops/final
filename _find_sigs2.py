import sys  
f=open(sys.argv[1],'r',encoding='utf-8')  
for i,l in enumerate(f,1):  
 if 'fit_first_stage' in l or 'fit_cf_cox' in l or 'def generate_data' in l:  
  print(sys.argv[1]+':'+str(i)+': '+l.rstrip())  
