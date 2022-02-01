import re
# Currently doesn't understand multiple insitutions for the first author
# Doesn't remove if the first author has > 3 names 
def Clean_PDF(element):
    score = 0 
    # Remove DOI
    X = re.search('^doi', element)
    temp = X != None
    score += temp
    # Remove Summary
    X = re.search('^summary',  element)
    temp = X != None
    score += temp
    # Remove Software
    X = re.search('^software', element)
    temp = X != None
    score += temp
    # Remove Review Repository Archive
    X = re.search('^• review', element)
    temp = X != None
    score += temp
    X = re.search('• repository', element)
    temp = X != None
    score += temp
    X = re.search('• archive', element)
    temp = X != None
    score += temp
    # Submitted - Published
    X = re.search('^submitted', element)
    temp = X != None
    score += temp
    X = re.search('^published', element)
    temp = X != None
    score += temp
    # Remove Licence CC-BY
    X = re.search('^license', element)
    temp = X != None
    score += temp
    X = re.search('^licence', element)
    temp = X != None
    score += temp
    # Remove Page Number
    if len(element.split(' ')) == 1:
        score += 1
    # Remove footer
    X = re.search('et al.,', element)
    temp = X != None
    score += temp
    # Remove Author Names
    if len(element.split(' ')) >1:
        if element.split(' ')[1][-1] == '1':
            score += 1
        if element.split(' ')[1][-2:] == '1,':
            score += 1
        if element.split(' ')[1][-2:] == '1\n':
            score += 1
    
    if len(element.split(' ')) >2:
        if element.split(' ')[2][-1] == '1':
            score += 1
        if element.split(' ')[2][-2:] == '1\n':
            score += 1
    # Remove Insitution Names     
    if element[0] == '1':
        score += 1
        if element[1] =='.':
            score = score - 1 # Attempting to include bullet points
    # Remove Future Woeks
    X = re.search('^future work', element)
    temp = X != None
    score += temp
    # Remove references
    X = re.search('^references', element) # Reference Title
    if element.split(' ')[0][-1] == ',':
        score += 1
    if element.split(' ')[0][-1] == '.':
        score += 1
        if len(element.split(' ')[0]) <3:
            score = score - 1 # Attempting to include bullet points
    if len(element.split(' ')) >2:
        if element.split(' ')[2][-1] == '.':
            score += 1
    # Remove Figures
    X = re.search('^figure', element) 
    temp = X != None
    score += temp
    
    return score