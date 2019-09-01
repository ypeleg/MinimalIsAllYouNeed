
from setuptools import setup, find_packages

setup(
    name='minimal_is_all_you_need',
    use_scm_version={"root": ".", "relative_to": __file__},
    setup_requires=['setuptools_scm'],
    description=('Minimal transformers library for Keras'),
    url='https://github.com/ypeleg/minimal_is_all_you_need',
    author='Yam Peleg',
    author_email='ypeleg2@gmail.com',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    keywords='development',
    packages=find_packages(where='.', exclude=['example']),
    install_requires=['Keras>=2.0.8', 'numpy'],
    tests_require=['pytest'],
    include_package_data=True,
)
