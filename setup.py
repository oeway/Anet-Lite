import sys

requirements = ['tensorflow==2.0.1', 'tensorflowjs==0.6.5', 'keras==2.2.4', 'git+https://www.github.com/keras-team/keras-contrib.git', 'pytest==3.5.0']

from setuptools import setup, find_packages
setup(name='anet-lite',
      version='0.3.0',
      description='Light-weight A-net for generic image-to-image translation.',
      url='http://github.com/oeway/Anet-Lite',
      author='Wei OUYANG',
      author_email='wei.ouyang@cri-paris.org',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      install_requires=requirements,
      zip_safe=False)
