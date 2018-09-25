import sys

requirements = ['tensorflow>=1.7.0', 'tensorflowjs>=0.1.2', 'keras>=2.1.4', 'keras_contrib', 'pytest==3.5.0']

from setuptools import setup, find_packages
setup(name='anet-lite',
      version='0.2.0',
      description='Light-weight A-net for generic image-to-image translation.',
      url='http://github.com/oeway/Anet-Lite',
      author='Wei OUYANG',
      author_email='wei.ouyang@cri-paris.org',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      install_requires=requirements,
      zip_safe=False)
