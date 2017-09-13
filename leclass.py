#unicode:utf-8
# class A:
# 	#classic class
# 	""" this is class A"""
# 	pass
# 	# __slots__={'x','y'}
# 	def test(self):
# 		#classic class test
# 		"""this is A.test()"""
# 		print('A class')

# class B(object):
# 	#new class
# 	""" this is class B"""
# 	__slots__={'x','y'}
# 	pass
# 	def test(self):
# 		#new class test
# 		"""this is B.test()"""
# 		print('B class')

# if __name__== '__main__':
# 	pass
# 	a=A()
# 	b=B()
# 	print('*'*10)
# 	print (dir(a))
# 	print('*'*10)
# 	print(dir(b))

# class Car:
# 	country=u'中国'
# 	def __init__(self,length,width,height,owner=None):
# 		self.owner = owner
# 		self.length = length
# 		self.width = width
# 		self.height = height
# 		self.country = 'china'

# if __name__ == '__main__':
# 	a = Car(1.2,1.4,1.5,'kuki')
# 	b = Car(2.2,2.5,2.6,'bob')
# 	print('a.owner:',a.owner)
# 	print('b.owner:',b.owner)
# 	print('a.country:',a.country)
# 	print('b.country:',b.country)

# 	b.country = 'America'

# 	print(a.country,'\n',b.country)
# 	print(Car.country)

# 	del a.country
# 	print(a.country)
# 	# del a.country
# 	# del Car.country
# 	# print(a.country)


# class Car:
# 	country = 'china'
# 	def __init__(self,length,width,height,owner=None):
# 		self.__owner = owner
# 		assert length>0,'length must larger than 0'
# 		self._length = length
# 		self._width = width
# 		self.height = height

# 	def getOwner(self):
# 		return self.__owner
# 	def setOwner(self,value):
# 		self.__owner = value

# 	def getLength(self):
# 		return self._length
# 	def setLength(self,value):
# 		assert value>0,'length must larger than 0'
# 		self._length = value

# if __name__ == '__main__':
# 	a=Car(1.2,1.4,1.5,'kuki')
# 	print(a.getOwner())


# class Car(object):
# 	country = 'china'
# 	__slots__ = ('owner','length','width','height','__dict__')
# 	def __init__(self,length,width,height,owner=None):
# 		self.owner = owner
# 		self.length = length
# 		self.width = width
# 		self.height = height

class Car(object):
    country = u'中国'
    __slots__=('length','width','height','owner','__dict__')
    
    def __init__(self, length, width, height, owner=None):
        self.owner = owner
        self.length = length
        self.width = width
        self.height = height
        
    #注意 只有当 __getattr__  __setattr__同时出现时才起作用
    def __getattr__(self,name):
        print ("__getattr__",name)
        assert name in self.__slots__, "Not have this attribute "+name
        return self.__dict__.get(name,None)

    def __setattr__(self,name,value):
        print ("__setattr__",name)
        assert name in self.__slots__, "Not have this attribute "+name
        
        if name!='owner':
            assert value>0, name+" must larger than 0"
        self.__dict__[name]=value
        
    def __delattr__(self,name):
        print ("__delattr__",name)
        assert name in self.__slots__, "Not have this attribute "+name
        if name=='owner':
            self.__dict__[name]=None

	# def __getattr__(self,name):
	# 	print('__getattr__',name)
	# 	return self.__dict__.get(name,None)

if __name__ == '__main__':
	a=Car(1.2,1.4,1.5,'kuki')

