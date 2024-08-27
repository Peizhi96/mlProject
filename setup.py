from setuptools import setup, find_packages

# write a function get a list of requirements from requirements.txt and remove the new line character
def get_requirements():
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()
        requirements = [req for req in requirements if req.strip() != '-e .']
    return requirements
    

setup(
    name='mlProject',
    version='0.0.1',
    author='Peizhi Yan',
    author_email='yanzhenyi123@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements()
)