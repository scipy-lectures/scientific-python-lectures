import sys

DEPENDENCIES = ['numpy', 'scipy', 'sklearn', 'matplotlib', 'nibabel']


def print_package_version(package_name, indent='  '):
    try:
        package = __import__(package_name)
        version = getattr(package, '__version__', None)
        package_file = getattr(package, '__file__', )
        provenance_info = '{0} from {1}'.format(version, package_file)
    except ImportError:
        provenance_info = 'not installed'

    print('{0}{1}: {2}'.format(indent, package_name, provenance_info))

if __name__ == '__main__':
    print('=' * 120)
    print('Python %s' % str(sys.version))
    print('from: %s\n' % sys.executable)

    print('Dependencies versions')
    for package_name in DEPENDENCIES:
        print_package_version(package_name)
    print('=' * 120)
