import os 
def makefolder(base='./',
               create_folder=False,
               precision=5,
               **kwargs,
               ):
  folder = base
  for key, value in kwargs.items():
    if value != None:
      if isinstance(value,float):
        folder += key + f'_{value:.{precision}f}/'
      else:
        folder += key + f'_{value}/'
  if create_folder:
    os.makedirs(folder,exist_ok=True)
  return folder

def get_scratch():
  wd = os.getcwd()
  text = wd
  keyword = "user=sacevedo"
  index = text.find(keyword)
  scratch = text[:index + len(keyword)] + '/scratch/sacevedo/'
  return scratch
