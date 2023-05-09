from distutils.core import setup
setup(
  name = 'musegan',         # How you named your package folder (MyLib)
  packages = ['musegan'],   # Chose the same as "name"
  version = '0.0.6',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'A pytorch implimentation of musegan by forked from: https://github.com/salu133445',   # Give a short description about your library
  author = 'clifford njoroge',                   # Type in your name
  author_email = 'cnjoroge925@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/cliffordkleinsr/musegan-pytorch',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/cliffordkleinsr/musegan-pytorch/archive/refs/tags/latest.tar.gz',    # I explain this later on
  keywords = ['pytorch', 'music', 'generative adverserial networks'],   # Keywords that define your package best
  install_requires=[            # SharedArray is needed but only works with linux
          'music21',
          'numpy',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Operating System :: MacOS :: MacOS X',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: POSIX',
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
  ],
)
