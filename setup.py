""" """
from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='dsfinesseur',
      version='0.1',
      description='The funniest joke in the world',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Topic :: Text Processing :: Linguistic',
        'Development Status :: 1 - Planning'
        'Intended Audience :: Developers'
        'Intended Audience :: Education'
        'Natural Language :: French'

      ],
      keywords='data science dsfinesseur ds finesse eda',
      url='http://github.com/jonitoh/dsfinesseur',
      author='TOH Ninsemou Jordan',
      author_email='njordant@hotmail.com',
      license='MIT',
      packages=['dsfinesseur'],
      install_requires=[
          'markdown',
      ],
      #test_suite='nose.collector',
      #tests_require=['nose', 'nose-cover3'],
      entry_points={
          'console_scripts': ['finesse-it=dsfinesseur.command_line:main'],
      },
      include_package_data=True,
      zip_safe=False)