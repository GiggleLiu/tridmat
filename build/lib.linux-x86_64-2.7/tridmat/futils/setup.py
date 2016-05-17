def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config=Configuration('futils',parent_package,top_path)
    config.add_extension('fmodule',['fysics.f90'],libraries=['lapack'])
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
