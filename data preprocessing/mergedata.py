class DBTable:
    def __init__(self,con,sql,batch_size):
        self.con = con
        self.sql = sql
        self.bs = batch_size

    def __iter__(self):
        print("__iter__called")
        return BufferTableIter(self)

class BufferTableIter():
    def __init__(self,dbTable):
        self.cursor = dbTable.con.cursor(buffered=True) #创建游标
        self.cursor.execute(dbTable.sql) #执行sql语句
        self.readCont = 0
        self.bs = dbTable.bs
        self.buf = [] # 用来存储当前批次的数据，初始化为空 
        self.idx = 0 # 当前批次数据（self.buf）指针，初始化为0
        
    def readBatch(self):
        if self.idx == len(self.buf):
            self.buf = self.cursor.fetchmany(size=self.bs)#从数据库中读取批次
            #更新指针
            self.readCont += len(self.buf)
            self.idx = 0


    def hasNext(self):
        self.readBatch()
        return self.idx < len(self.buf)

    def readNext(self):
        self.readBatch()
        if self.idx < len(self.buf):
            line = self.buf[self.idx]
            self.idx += 1
            return line

    def __iter__(self):
        return self

    def __next__(self):
        if self.hasNext():
            return self.readNext()
        else:
            raise StopIteration
