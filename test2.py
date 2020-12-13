#%%
class DataTest:
    def __init__(self,id,address):
        self.id=id
        self.address=address
        self.d={self.id:1,
                self.address:"192.168.1.1"}
        
    def __getitem__(self, key):
        if not isinstance(key, int):
            raise Exception("Index must be an integer!")
        print(key)
        return key
    
 
data=DataTest(1,"192.168.2.11")
data[1:2]
# %%
