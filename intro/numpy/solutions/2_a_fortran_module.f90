subroutine some_function(n, a, b)
  integer :: n
  double precision, dimension(n), intent(in) :: a
  double precision, dimension(n), intent(out) :: b
  b = a + 1
end subroutine some_function
