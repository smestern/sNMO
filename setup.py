from setuptools import setup, find_packages


packages = find_packages()
print(packages)
setup(
    name='sNMO',
    version='0.1',
    packages=packages,
    package_dir={'sNMO': '.'},
    install_requires=[
        # List your package's dependencies here #TODO
    ],
    
)
