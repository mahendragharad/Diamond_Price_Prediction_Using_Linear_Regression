from setuptools import find_packages , setup 
from typing import List 

# To avoid it from installation 
HYPEN_E_DOT = '-e .'

# Taking the filename in str format and returning the list of data
def get_requirements(file_name : str ) -> List[str] :
    requirements = [] 
    with open(file_name) as file :
        requirements = file.readline()
        requirements = [ req.replace("\n" , " ") for req in requirements ]
        
        # To remove -e . 
        if HYPEN_E_DOT in requirements :
            requirements.remove(HYPEN_E_DOT)

    return requirements




setup (

    name = "Regression Project" ,
    version = "0.0.1",
    author = "Mahendra",
    author_email= "mahendragharad5@gmail.com",
    installs_requires = get_requirements("requirements.txt") ,
    packages = find_packages()
)