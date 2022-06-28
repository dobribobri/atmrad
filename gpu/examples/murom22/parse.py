
from cpu import ar
import os
from bs4 import BeautifulSoup
import re
import dill


DATA = {}
folder = 'Dolgoprudnyj'
htmls = os.listdir(folder)
for html in htmls:
    print(html)
    with open(os.path.join(folder, html), 'r') as f:
        contents = f.read()
        soup = BeautifulSoup(contents, "html.parser")
        all_h2 = soup.findAll('h2')
        measurements = []
        for h2 in all_h2:
            h2 = str(h2)[4:-5]
            n = re.split(r' ', h2)
            year = n[-1]
            month = n[-2]
            day = n[-3]
            meas = n[-4]
            measurements.append((year, month, day, meas))
        all_pre = soup.findAll('pre')[::2]
        if len(measurements) != len(all_pre):
            continue
        for i, pre in enumerate(all_pre):
            pre = re.split('\n', str(pre))[4:]
            T, P, rel, alt = [], [], [], []
            for line in pre:
                num = [float(e) for e in re.split(r'[ \t]', re.sub(r'[^0-9.\- \t]', '', line)) if e]
                valid = len(num) == 11
                if valid:
                    # print(num)
                    P.append(num[0])
                    alt.append(num[1] / 1000)
                    T.append(num[2])
                    rel.append(num[4])
            T, P, rel, alt = [ar.op.as_tensor(a) for a in [T, P, rel, alt]]
            DATA[measurements[i]] = (T, P, rel, alt)
with open('Dolgoprudnyj.dump', 'wb') as dump:
    dill.dump(DATA, dump, recurse=True)
