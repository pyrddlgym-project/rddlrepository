import os 
from rddlrepository.core.manager import RDDLRepoManager
 

def main():
    content = []    
    content.append(r'\hline')
    content.append(r'Context & Problem & Description & Instances \\')
    
    manager = RDDLRepoManager(rebuild=True)
    for context in manager.list_contexts():
        content.append(r'\hline')
        for problem in manager.list_problems_by_context(context):
            info = manager.get_problem(problem)
            instances = ', '.join(info.list_instances())
            row = [context.replace('_', '\\_'),
                   problem.replace('_', '\\_'),
                   info.desc, instances]
            content.append(' & '.join(row) + r'\\')
    content.append(r'\hline')
    
    table = (
        r'\begin{tabular}{ |c|c|c|c| }' + '\n'
        +'\n'.join(content) + '\n'
        +r'\end{tabular}'
    )        
    doc = (
        r'\documentclass{standalone}' + '\n'
        r'\begin{document}' + '\n'
        +table + '\n'
        +r'\end{document}'
    )
    with open("domains.tex", "w") as latex_file:
        latex_file.write(doc)
    os.system("pdflatex domains.tex")


if __name__ == '__main__':
    main()
