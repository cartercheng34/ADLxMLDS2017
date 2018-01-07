import re

def parse(tags_path):
    with open('./trim.txt' , 'w') as f:
        for line in open(tags_path):
            id = line.split(',')[0]
            #print(id + ',' , file=f,end='')
            tmp = re.findall('\\b[a-zA-Z]*\\s+[a-zA-Z]*\\s+hair\\b' , line.split(',')[1])                        
            tmp2 = re.findall('\\b[a-zA-Z]*\\s+[a-zA-Z]*\\s+eyes\\b' , line.split(',')[1])
            if len(tmp) != 0 or len(tmp2) != 0:
                print(id + ',' , file=f,end='')
                for i in range(len(tmp)):
                    print(tmp[i].strip() + ' ', file=f , end='')
                
                for j in range(len(tmp2)):
                    print(tmp2[j].strip() + ' ', file=f , end='')
                print('' , file=f)
            #for i in range(len(tmp2)):
                #print(tmp2[i].strip() , file=f)
            #tmp = re.findall('\\b[a-zA-Z]*\\s+[a-zA-Z]*\\s+hair\\b' , line.split(',')[1])
            #print('' , file=f)



tags_path = './tags_clean.csv'
parse(tags_path)
"""
with open('./tags.txt' , 'w') as f:
	for i in range(image_num):
		img = Image.open(os.path.join('faces', str(i) + '.jpg'))
		a = illust2vec.estimate_plausible_tags([img] , threshold=0.1)

		#pdb.set_trace()
		f.write(str(i) + ',' )
		for k,v in a[0].items():
			if k == 'general':
				for j in range(len(v)):
					if 'hair' in v[j][0] or 'eyes' in v[j][0]:
						pdb.set_trace()
						print (v[j][0]+' ' , file=f, end='')
"""