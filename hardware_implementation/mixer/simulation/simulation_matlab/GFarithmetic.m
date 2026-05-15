%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%{
function r = deflate(LUT, a, b)
	n = length(a);
	m = length(b);
	if (n == m)
		r = zeros(n,1);
		for i = 1 : n
			r(i) = LUT(a(i) + 1, b(i) + 1);
		endfor
		r = reshape(r, size(a));
	elseif (n == 1)
		r = zeros(m,1);
		for i = 1 : m
			r(i) = LUT(a + 1, b(i) + 1);
		endfor
		r = reshape(r, size(b));
	elseif (m == 1)
		r = zeros(n,1);
		for i = 1 : n
			r(i) = LUT(a(i) + 1, b + 1);
		endfor
		r = reshape(r, size(a));
	else
		error("so nicht!");
	endif	
%	r = uint8(r);
endfunction

function r = GF2add(a, b)
	persistent LUT = ...
	[
		0	1;
		1	0
	];
	r = deflate(LUT, a, b);
endfunction

function r = GF2sub(a, b)
	persistent LUT = [0 1];
	r = GF2add(a, deflate(LUT, 0, b));
endfunction

function r = GF2mul(a, b)
	persistent LUT = ...
	[
		0	0;
		0	1
	];
	r = deflate(LUT, a, b);
endfunction

function r = GF2div(a, b)
	persistent LUT = [0 1];
	r = GF2mul(a, deflate(LUT, 0, b));
endfunction
%}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function r = GF2_add(a, b)
	r = rem(a + b, 2);
%	r = uint8(rem(a + b, 2));
%	r = bitxor(a, b);
%	r = bitxor(uint8(a), uint8(b));
	% mod version seems to be a bit faster in some situations, most likely because of the typecasts.
	% it is safe in this manner as long as a and b are non-negative.
endfunction

% GF2_sub() didn't work right with uint8 up to mixerVersion 1.0 (fixed with 1.1)
function r = GF2_sub(a, b)
	r = bitxor(a, b);
%	r = bitxor(uint8(a), uint8(b));
%	r = uint8(mod(int8(a) - int8(b), 2));
	% xor version seems to be a bit faster for some reason, most likely because of the typecasts.
	% it is safe in this manner because bitxor() interprets all values as integers (see help/doc).
endfunction

function r = GF2_mul(a, b)
	r = rem(a .* b, 2);
endfunction

function r = GF2_mmul(a, b)
	r = uint8(rem(double(a) * double(b), 2));
	% attention: 8 x 8 <= 16 bit, 53 (precision of double) - 16 = 37 bit -> save with vector size <= 2^37
	% if you need greater size: implement block processing with intermediate modulo operations
endfunction

function r = GF2_div(a, b)
	r = rem(a ./ b, 2);
endfunction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function r = GFp_add(a, b, p)
	r = uint8(rem(uint16(a) + uint16(b), p));
endfunction

function r = GFp_sub(a, b, p)
	r = uint8(mod(int16(a) - int16(b), p));
	% attention: don't use int8 (although the result will fit into 8 bit for sure) because int8(>127) = 127
endfunction

function r = GFp_mul(a, b, p)
	r = uint8(rem(uint16(a) .* uint16(b), p));
endfunction

function r = GFp_mmul(a, b, p)
	r = uint8(rem(double(a) * double(b), p));
	% attention: 8 x 8 <= 16 bit, 53 (precision of double) - 16 = 37 bit -> save with vector size <= 2^37
	% if you need greater size: implement block processing with intermediate modulo operations
endfunction

function r = GFp_div(a, b, p)

	if (!isscalar(b))
		error("operand sizes do not match");
	endif
	if (b == 0)
		error("division by zero");
	endif

	persistent LUT = [];

	if (rows(LUT) != p)

		LUT = zeros(p);

		for i = 2 : rows(LUT)
			LUT(i,:) = GFp_mul(i - 1, 0 : p - 1, p);
			[~, LUT(i,:)] = sort(LUT(i,:));
		endfor
		LUT -= 1;

	endif

	r = reshape(LUT(uint32(b)+1, uint32(a)+1), size(a));

endfunction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function r = GF2n_add(a, b, n)

	r = bitxor(a, b);
%	r = bitxor(uint8(a), uint8(b));
	% xor version without typecasts is a bit faster.
	% it is safe in this manner because bitxor() interprets all values as integers (see help/doc).
	
%	aa = reshape(bitunpack(vec(uint8(a))), 8, [])(1:n, :);
%	bb = reshape(bitunpack(vec(uint8(b))), 8, [])(1:n, :);
%
%	s = rem(aa + bb, 2);
%
%	r = reshape((2 .^ (0 : n - 1)) * s, max(size(a), size(b)));

endfunction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function r = GF2n_sub(a, b, n)

	r = bitxor(a, b);
%	r = bitxor(uint8(a), uint8(b));
	% xor version without typecasts is a bit faster.
	% it is safe in this manner because bitxor() interprets all values as integers (see help/doc).
	
%	aa = reshape(bitunpack(vec(uint8(a))), 8, [])(1:n, :);
%	bb = reshape(bitunpack(vec(uint8(b))), 8, [])(1:n, :);
%
%	s = mod(aa - bb, 2);
%
%	r = reshape((2 .^ (0 : n - 1)) * s, max(size(a), size(b)));

endfunction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [r, LUT] = GF2n_mul(a, b, n)

	% don't support vector-vector product as long as there is no need (and no clear definition) for it
	if (!isscalar(a))
		if (isscalar(b))
			t = a; a = b; b = t;
		else
			error("operand sizes do not match");
		endif
	endif
	
	persistent LUT = [];

	if (rows(LUT) != 2^n)

		polynomial = {
			[1 1],
			[1 1 1],
			[1 1 0 1],
			[1 1 0 0 1],
			[1 0 1 0 0 1],
			[1 1 0 0 0 0 1],
			[1 1 0 0 0 0 0 1],
			[1 1 0 0 0 1 1 0 1]
		}{n};

		LUT = zeros(2^n);

		for i = 1 : rows(LUT)
			for k = i : rows(LUT)

				ii = bitunpack(uint8(i - 1))(1 : n);
				kk = bitunpack(uint8(k - 1))(1 : n);

				p = rem(conv(ii, kk), 2);
				
				while (length(p) > n)
					p(end - n : end) = bitxor(p(end - n : end), p(end) * polynomial);
					p(end) = [];
				endwhile

				LUT(i,k) = p * (2 .^ (0 : n - 1)');

			endfor
		endfor

		LUT += triu(LUT, 1)';
		
	endif

	r = reshape(LUT(uint32(a)+1, uint32(b)+1), size(b));

endfunction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function r = GF2n_mmul(a, b, n)

	r = uint8(0)(ones(rows(a), columns(b)));

	if (sum(size(a)) >= sum(size(b)))
		for i = 1 : columns(b)
			for k = 1 : rows(b)
				r(:,i) = GF2n_add(r(:,i), GF2n_mul(b(k,i), a(:,k), n), n);
			endfor
		endfor
	else
		for i = 1 : rows(a)
			for k = 1 : columns(a)
				r(i,:) = GF2n_add(r(i,:), GF2n_mul(a(i,k), b(k,:), n), n);
			endfor
		endfor
	endif

endfunction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function r = GF2n_div(a, b, n)

	if (!isscalar(b))
		error("operand sizes do not match");
	endif
	if (b == 0)
		error("division by zero");
	endif

	persistent LUT = [];

	if (rows(LUT) != 2^n)

		[~, LUT] = GF2n_mul(0, 0, n);

		for i = 2 : rows(LUT)
			[~, LUT(i,:)] = sort(LUT(i,:));
		endfor
		LUT -= 1;
	
	endif

	r = reshape(LUT(uint32(b)+1, uint32(a)+1), size(a));

endfunction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function d = GFunpack(x, fieldSize)

	[p,n] = factor(fieldSize);
	if (length(p) != 1)
		error("invalid fieldSize\np = %sn = %s", disp(p), disp(n));
	endif

	d = bitunpack(x);
	
	if (fieldSize == 2)
		return;
	endif

	% don't use datatypes which might lead to errors 
	% caused by accuracy issues when casting to double
	if (length(d) > 32)
		error("type %s is unsupported right now", class(x));
	endif

	t = (2 .^ (0 : length(d) - 1)) * vec(double(d));
	l = ceil(length(d) / log2(fieldSize));

	d = [];
	for k = l - 1 : -1 : 0
		d(k + 1) = fix(t / fieldSize ^ k);
		t -= d(k + 1) * fieldSize ^ k;
	endfor

	if (rows(x) > 1)
		d = vec(d);
	endif
	
endfunction
	
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function x = GFpack(d, fieldSize, class_)

	[p,n] = factor(fieldSize);
	if (length(p) != 1)
		error("invalid fieldSize\np = %sn = %s", disp(p), disp(n));
	endif

	if (max(d) >= fieldSize)
		error("invalid input data");
	endif
	
	x = (fieldSize .^ (0 : length(d) - 1)) * vec(double(d));
	x = bitpack(bitunpack(uint64(x)), class_)(1);
	
endfunction
	
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% plausibility checks
function GFpn_check(p, n)

	if (n > 1)
		if (p != 2)
			error("unsupported parameter combination");
		endif
		GFadd = @(a,b) GF2n_add(a,b,n);
		GFsub = @(a,b) GF2n_sub(a,b,n);
		GFmul = @(a,b) GF2n_mul(a,b,n);
		GFdiv = @(a,b) GF2n_div(a,b,n);
		GFmmul = @(a,b) GF2n_mmul(a,b,n);
	elseif (p > 2)
		GFadd = @(a,b) GFp_add(a,b,p);
		GFsub = @(a,b) GFp_sub(a,b,p);
		GFmul = @(a,b) GFp_mul(a,b,p);
		GFdiv = @(a,b) GFp_div(a,b,p);
		GFmmul = @(a,b) GFp_mmul(a,b,p);
	else
		GFadd = @(a,b) GF2_add(a,b);
		GFsub = @(a,b) GF2_sub(a,b);
		GFmul = @(a,b) GF2_mul(a,b);
		GFdiv = @(a,b) GF2_div(a,b);
		GFmmul = @(a,b) GF2_mmul(a,b);
	endif

	maxValue = p ^ n - 1;

	addLUT = zeros(p^n);
	for i = 1 : rows(addLUT)
		addLUT(i,:) = GFadd(i-1, 0 : maxValue);
	endfor
	addLUT

	if (max(max(addLUT)) > maxValue)
		warning("add is not closed");
	endif
	if (!issymmetric(addLUT))
		warning("add is not commutative");
	endif
	if (addLUT(1,:) != (0 : maxValue))
		warning("add neutral element incorrect");
	endif
	for i = 1 : rows(addLUT)
		if (length(unique(addLUT(i,:))) != rows(addLUT))
			warning("add (%d) is not complete", i);
		endif
	endfor

	mulLUT = zeros(p^n);
	for i = 1 : rows(mulLUT)
		mulLUT(i,:) = GFmul(i-1, 0 : maxValue);
	endfor
	mulLUT

	if (max(max(mulLUT)) > maxValue)
		warning("mul is not closed");
	endif
	if (!issymmetric(mulLUT))
		warning("mul is not commutative");
	endif
	if (mulLUT(1,:) != zeros(1,rows(mulLUT)))
		warning("mul zero element incorrect");
	endif
	if (mulLUT(2,:) != (0 : maxValue))
		warning("mul neutral element incorrect");
	endif
	for i = 2 : rows(mulLUT)
		if (length(unique(mulLUT(i,:))) != rows(mulLUT))
			warning("mul (%d) is not complete", i);
		endif
	endfor

	for i = 0 : maxValue
		t = GFsub(i, 0 : maxValue);
		if (unique(GFadd(t, 0 : maxValue)) != i)
			warning("sub (%d) is incorrect", i);
		endif
	endfor

	for i = 1 : maxValue
		t = GFdiv(0 : maxValue, i);
		if (GFmul(t, i) != (0 : maxValue))
			warning("div (%d) is incorrect", i);
		endif
	endfor

	A = randi(p^n - 1, 3);
	x = randi(p^n - 1, 3, 1);

	Ax = GFmmul(A,x);
	y = uint8(0)(ones(rows(A), 1));
	for i = 1 : length(x)
		y = GFadd(y, GFmul(x(i), A(:,i)));
	endfor
	if (!isequal(y, Ax))
		warning("mmul is incorrect");
		A, x, Ax, y
	endif

	x = x';
	xA = GFmmul(x,A);
	y = uint8(0)(ones(1, columns(A)));
	for i = 1 : length(x)
		y = GFadd(y, GFmul(x(i), A(i,:)));
	endfor
	if (!isequal(y, xA))
		warning("mmul is incorrect");
		A, x, xA, y
	endif

endfunction

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
