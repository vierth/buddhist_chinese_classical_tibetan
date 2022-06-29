import csv, hanziconv, os, platform, pickle
from multiprocessing import Pool
from pavdhutils.cleaning import clean

# this function loads a csv file and returns it as a list of the rows. the 
# first row will be the header
def load_csv(filepath, simplified=False,delim=","):
    with open(filepath, 'r', encoding='utf8') as rf:
        reader = csv.reader(rf, delimiter=delim)
        data = [row for row in reader]
        
        if simplified:
            s_d = []
            for d in data:
                s_d.append([hanziconv.HanziConv.toSimplified(i) for i in d])
            data = s_d     
    return data    

# load a text file as a string
def load_text(filepath, simplified=False):
    with open(filepath, 'r', encoding='utf8') as rf:
        text = rf.read()
        if simplified:
            text = hanziconv.HanziConv.toSimplified(text)
    return text

# load a directory, return a list of contents of files
def load_dir(dir_path, simplified=False):
    labels, texts = [], []
    for root, dirs, files in os.walk(dir_path):
        print(f"Loading {len(files)} files in {root}")
        for i,fname in enumerate(files):
            if i % len(files)/4 == 0:
                print("One quarter finished")
            texts.append(load_text(os.path.join(root, fname),simplified))
            labels.append(fname)

    return labels, texts

def parallel_load_text(filepath, simplified=True, clean_text=True):
    with open(filepath, 'r', encoding='utf8') as rf:
        text = rf.read()
        if simplified:
            text = hanziconv.HanziConv.toSimplified(text)
        if clean_text:
            text = clean(text)
    label = os.path.split(filepath)[-1]
    
    return (label,text)

def parallel_load_dir(dir_path):
    all_res = []
    for root, dirs, files in os.walk(dir_path):
        full_paths = [os.path.join(root,f) for f in files]
        with Pool(60) as p:
            results = p.map(parallel_load_text, full_paths) 
        all_res.extend(results)
    all_res.sort()
    return zip(*all_res)

# turn a list of lists into a dictionary using one of the values in the list as
# a key
def csv_to_dict(lists, key_index=0, header=True, return_meta=False):
    res_dict = {}
    if header:
        header_dict = {}
        header_dict[lists[0][key_index]] = [v for i,v in enumerate(lists[0]) 
                                            if i != key_index]
        lists = lists[1:]
        
    for l in lists:
        if l[key_index] not in res_dict:
            res_dict[l[key_index]] = [[v for i,v in enumerate(l) if i != key_index]]
        else:
            res_dict[l[key_index]].append([v for i,v in enumerate(l) if i != key_index])
    if return_meta:
        return res_dict, header_dict
    else:
        return res_dict

# a simple find all function. returns a list of results or None
def find_all(term, text):
    results = []
    if term in text:
        location = text.find(term)
        while location != -1:
            results.append(location)
            location = text.find(term, location + 1)
        return results
    else:
        return None
        
# set matplotlib font options depending on operating system
def set_mpl(lang='zh'):
    import matplotlib
    if lang == 'zh':
        if platform.system() == "Windows":
            font_name = "SimHei"
        elif platform.system() == "Darwin":
            font_name == "STHeiti"
    else:
        font_name = "Consolas"
    matplotlib.rcParams['font.family']=font_name
    matplotlib.rcParams['axes.unicode_minus']=False