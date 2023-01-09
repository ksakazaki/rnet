from setuptools import setup
import rnet

if __name__ == '__main__':
    setup(
        name='RNet',
        version=rnet.__version__,
        author=rnet.__author__,
        author_email='kotasakazaki@sophia.ac.jp',
        packages=['rnet']
        )
