!This is A Fortran Library to boost python code.
!The topic is matrix linear algebra tricks

!this is a function that update det(inv(G)) with only 1-line changed.
!N: the dimension of the matrix
!Ginv: the old inversion of G
!G: the old G
!ith: the row(column index)
!vec: the replace vector
!crow: change row if true else column
    !if crow=T, the ith-row of Ginv => vec
    !if crow=F, the ith-column of Ginv => vec
!ratio: the ratio.
    !ratio = det(new Ginv) / det(old Ginv)
!ratioonly: get the ratio only if false.
    !if ratioonly=T, only ratio is calculated
    !if ratioonly=F, Ginv and G are also updated
subroutine fdyson(N,Ginv,G,ith,vec,crow,ratio,ratioonly)
    implicit none
    integer,intent(in) :: N,ith
    complex*16,intent(inout) :: Ginv(N,N),G(N,N)
    complex*16,intent(out) :: ratio
    complex*16,intent(in) :: vec(N)
    complex*16 :: work(N,N),Gii
    logical,intent(in) :: ratioonly,crow
    !f2py intent(out) :: ratio
    !f2py intent(in) :: N,ith,ratioonly,crow,vec
    !f2py intent(inout) :: Ginv,G

    if(crow)then
        ratio = sum( vec*G(:,ith) )
    else
        ratio = sum( G(ith,:)*vec )
    end if
    if(ratioonly)return

    !update Ginv
    if(crow)then; Ginv(ith,:)=vec; else; Ginv(:,ith)=vec; end if
  
    !prepare auxiliary green's func: work
    Gii=G(ith,ith)

    !in case Gii=0 or ratio is too small, G is renewed de fort
    if(abs(Gii)<1.e-10 .or. abs(ratio)<1.e-10)then
        G = Ginv; call zinv(N,G); return
    end if

    work = G - matmul( G(:,ith:ith), G(ith:ith,:) ) / Gii
    !notice that work(ith,:)=work(:,ith)=0

    !save corrected G(ith,ith)
    Gii=Gii/ratio

    !calculate corrected G(:,ith) and G(ith,:) using work and Ginv
    G(:,ith:ith) = - matmul(work,Ginv(:,ith:ith))*Gii
    G(ith:ith,:) = - matmul(Ginv(ith:ith,:),work)*Gii

    !recover G(ith,ith), which was erased by work in the above 
    G(ith,ith)=Gii

    !get all elements of corrected G
    G = matmul(G(:,ith:ith),G(ith:ith,:))/Gii + work     
    return
end subroutine fdyson

!check dyson
!N: the dyson-update times
subroutine fcheckdyson(N)
    implicit none
    integer :: iseed,ith,ispin,i,j,k,ithrow,jthcol
    integer,intent(in) :: N
    complex*16 :: ratio
    complex*16,dimension (N) :: vec,col,row
    complex*16,dimension (N,N) :: G,Ginv,work
    complex*16  :: det,det1
    !f2py intent(in) :: N

    iseed=97235

    do j=1,N; do i=1,N
        Ginv(i,j)=cmplx( ran(iseed)-0.5, ran(iseed)-0.5 )
    end do; end do
    G=Ginv; call zinv(N,G); call zdet(N,Ginv,det)

    do k=1,100
        do ith=1,N
            row = cmplx( ran(iseed)-0.5, ran(iseed)-0.5 )
            col = cmplx( ran(iseed)-0.5, ran(iseed)-0.5 )
            ithrow=int(N*ran(iseed))+1; jthcol=int(N*ran(iseed))+1
            call fdyson2(N,ithrow,row,jthcol,col,Ginv,G,ratio,.false.)
            call zdet(N,Ginv,Det1)
            if(abs(det1/det-ratio)>1.e-6)then
                print*,ratio,det1/det
                stop 'rowcol failed'
            end if
            Det = Det1
            work=matmul(G,Ginv)
            do i=1,N; do j=1,N
                if(i==j)then; if(abs(work(i,i)-1)>1.e-7)print*,work(i,i); end if
                if(i/=j)then; if(abs(work(i,j))>1.e-7)print*,work(i,j); end if
            end do; end do
        end do
    end do

    do ith=1,N
        do ispin=1,2
            do i=1,N; vec(i) = cmplx( ran(iseed)-0.5, ran(iseed)-0.5 ); end do
            if(ispin==1) then
                call fdyson(N,Ginv,G,ith,vec,.true.,ratio,.false.)
            else
                call fdyson(N,Ginv,G,ith,vec,.false.,ratio,.false.)
            endif
            call zdet(N,Ginv,Det1)
            if(abs(det1/det-ratio)>1.e-6)then
                print*,ratio,det1/det
                stop 'dyson failed'
            end if
            det=det1
        end do
    end do
    print*,'dyson ok'
    return
end subroutine fcheckdyson

!similar to dyson, but change row and col at the same time.
subroutine fdyson2(N,ithrow,row,jthcol,col,Ginv,G,ratio,ratioonly)
    implicit none
    integer,intent(in) :: N,ithrow,jthcol
    logical,intent(in) :: ratioonly
    complex*16,intent(out) :: ratio
    complex*16,dimension(N),intent(in) :: row,col
    complex*16,dimension(N,N),intent(inout) :: Ginv,G

    integer,dimension(N) :: maprow,mapcol
    integer :: i,j,i0,j0,irow,jcol,icol,jrow
    complex*16,dimension (N) :: G0i,Gi0
    complex*16 :: Mij,Mji,G00
    !f2py intent(out) :: ratio
    !f2py intent(inout) :: Ginv,G
    !f2py intent(in) :: N,ithrow,jthcol,ratioonly,row,col

    if(ithrow>N.or.ithrow<1)stop 'invalid row label'
    if(jthcol>N.or.jthcol<1)stop 'invalid col label'

    !row labels are mapped so that ithrow --> first row
    maprow(1) = ithrow 
    do i=1,ithrow-1; maprow(i+1) = i; end do
    do i=ithrow+1,N; maprow(i) = i; end do

    !col labels are mapped so that jthcol --> first column
    mapcol(1) = jthcol
    do i=1,jthcol-1; mapcol(i+1) = i; end do
    do i=jthcol+1,N; mapcol(i) = i; end do

    !(row,col) mapped to (0,0)
    i0 = maprow(1); j0 = mapcol(1)

    !notice: row and col obey different mapping in G and Ginv:
    !for G(i,j), i <--> mapcol(i),  j <--> maprow(j),
    !for Ginv(i,j), i <--> maprow(i), j <--> mapcol(j).

    ratio = col(ithrow) * G(j0,i0)
    !assuming col overwrites row at the crossing position
   
    do j = 2, N
        jrow = maprow(j)
        do i = 2, N
            icol = mapcol(i)
            Mij =  G(icol,jrow)*G(j0,i0) - G(icol,i0) * G(j0,jrow)
            ratio = ratio - row(icol) * Mij * col(jrow)
        end do
    end do

    if(ratioonly) return

    !update Ginv
    Ginv(ithrow,:) = row;  Ginv(:,jthcol) = col

    if(abs(ratio) < 1.e-10 .or. abs(G(j0,i0)) < 1.e-10 )then
        G = Ginv; call zinv(N,G); return
    end if
  
    G00 = G(j0,i0) / ratio
    Gi0 = G(:,i0); G0i = G(j0,:)        

    !update G(:,i0) and G(i0,:)
    do j = 2,N; jrow = maprow(j); jcol = mapcol(j)  
        G(jcol,i0) = 0; G(j0,jrow) = 0
        do i = 2,N;  irow = maprow(i); icol = mapcol(i)
            Mij = G(icol,jrow)*G(j0,i0) - Gi0(icol) * G0i(jrow) 
            Mji = G(jcol,irow)*G(j0,i0) - Gi0(jcol) * G0i(irow) 
            G(jcol,i0) = G(jcol,i0) - Mji*col(irow)/ratio
            G(j0,jrow) = G(j0,jrow) - row(icol)*Mij/ratio
        end do
    end do

    !update G(i,j) for i/=j0 and j/=i0
    do j = 2,N;   jrow = maprow(j)
    do i = 2,N;   icol = mapcol(i)
        Mij = G(icol,jrow)*G(j0,i0) - Gi0(icol) * G0i(jrow)      
        G(icol,jrow) = Mij/G(j0,i0) + G(icol,i0) * G(j0,jrow)/G00
    end do; end do

    G(j0,i0) = G00
    return
end subroutine fdyson2

!Turn inptr of csr matrix to x-indices
!n: length of ptr
!ndata: length of indices
!indptr: indptr of csr matrix.

!return -> xinds: x indices
subroutine fcsr_xindices(ptr,n,ndata,xinds)
    implicit none
    integer,intent(in) :: n,ndata
    integer,intent(in),dimension(n) :: ptr
    integer,intent(out),dimension(ndata) :: xinds
    integer :: row
    
    !f2py intent(in) :: n,ptr,ndata
    !f2py intent(out) :: xinds
    do row=1,n-1
        xinds(ptr(row)+1:ptr(row+1))=row-1
    enddo
    xinds(ptr(row)+1:ndata)=n-1
end subroutine fcsr_xindices

!Turn x-indices of to indptrs
!n: the matrix dimension +1.
!ndata: length of x indices
!indptr: indptr of csr matrix.

!return -> xinds: x indices
subroutine fcsr_ptrindices(xinds,ptr,n,ndata)
    implicit none
    integer,intent(in) :: n,ndata
    integer,intent(in),dimension(ndata) :: xinds
    integer,intent(out),dimension(n) :: ptr
    integer :: row,datai
    
    !f2py intent(in) :: n,ndata,xinds
    !f2py intent(out) :: ptr
    ptr(1)=0
    datai=1
    do row=2,n
        if(xinds(datai)>=row-1) then
            ptr(row)=datai-1
        else
            do while(xinds(datai)<row-1 .and. datai<=ndata)
                datai=datai+1
            enddo
            ptr(row)=datai-1
        endif
    enddo
end subroutine fcsr_ptrindices

!inversion of matrix A
SUBROUTINE zinv(N,A)
    IMPLICIT NONE
    INTEGER N
    COMPLEX*16 A(N,N), WORK(2*N)
    INTEGER INFO, IPIV(N), LWORK
    !f2py intent(inout) :: A
    !f2py intent(in) :: N
    IF(N<=0)STOP 'INVALID DIM @ zinv'
    IF(N==1)THEN; A=1/A; RETURN; END IF
    CALL ZGETRF(N, N, A, N, IPIV, INFO)
    IF(INFO/=0)PRINT*, 'LU@INVERSE DETECTED A SINGULARITY'
    LWORK=2*N
    CALL ZGETRI(N, A, N, IPIV, WORK, LWORK, INFO)
    IF(INFO/=0)STOP 'zinv FAILED'
    RETURN
END SUBROUTINE zinv

!determinant of matrix A
SUBROUTINE zdet(N,A,RES)
    IMPLICIT NONE
    INTEGER,intent(in) :: N
    INTEGER :: I,INFO
    INTEGER :: IPVT(N)
    COMPLEX*16,intent(in) :: A(N,N)
    COMPLEX*16 :: B(N,N)
    COMPLEX*16,intent(out) :: RES
    !f2py intent(inout) :: A,RES
    !f2py intent(in) :: N
    IF(N<=0)STOP 'INVALID DIM @ zdet'
    IF(N==1)THEN; RES=A(1,1); RETURN; END IF
    B=A;  CALL ZGETRF(N,N,B,N,IPVT,INFO)
    IF(INFO/=0)THEN; RES=0; RETURN; END IF          
    INFO=1; RES=0
    DO I=1, N
        IF(IPVT(I)/=I)INFO=-INFO
        RES=RES+LOG(B(I,I))
    END DO
    RES=EXP(RES); IF(INFO<0)RES=-RES
    RETURN
END SUBROUTINE zdet

!get the sequences of twisted LU decomposition for tridiagonal `Block` matrix.
!A=LU.
!reference -> http://dx.doi.org/10.1016/j.amc.2005.11.098

!input:
!al/bl/cl
!the lower diagonal/diagonal/upper diagonal part of matrix.

!return:
!(ll1,ll1,ll1,ul2,ul2,ul2,hl3,hl3,hl3,invhl1,invhl2,invhl3), they are sequences defining L,U
subroutine fget_tlu_seqsn(al,bl,cl,ll1,ll2,ll3,ul1,ul2,ul3,hl1,hl2,hl3,invhl1,invhl2,invhl3,n,p)
    implicit none
    integer,intent(in) :: n,p
    complex*16,intent(in),dimension(n,p,p) :: bl
    complex*16,intent(in),dimension(n-1,p,p) :: al,cl
    complex*16,intent(out),dimension(n,p,p) :: hl1,hl2,hl3,invhl1,invhl2,invhl3
    complex*16,intent(out),dimension(n-1,p,p) :: ll1,ll2,ll3,ul1,ul2,ul3
    complex*16,dimension(p,p) :: bi,hi,ui,invhi
    integer :: i
    
    !f2py intent(in) :: n,p,al,bl,cl
    !f2py intent(out) :: ll1,ll2,ll3,ul1,ul2,ul3,hl1,hl2,hl3,invhl1,invhl2,invhl3

    !sequence for i < j
    ll1=al
    do i=1,n-1
        bi=bl(i,:,:)
        hi=bi
        if(i/=1) then
            hi=hi-matmul(ll1(i-1,:,:),ui)
        endif
        invhi=hi
        call zinv(p,invhi)
        hl1(i,:,:)=hi
        invhl1(i,:,:)=invhi
        ui=matmul(invhi,cl(i,:,:))
        ul1(i,:,:)=ui
    enddo
    !sequence for i > j
    ll2=cl
    do i=n,2,-1
        hi=bl(i,:,:)
        if(i/=n) then
            hi=hi-matmul(ll2(i,:,:),ui)
        endif
        invhi=hi
        call zinv(p,invhi)
        hl2(i,:,:)=hi
        invhl2(i,:,:)=invhi
        ui=matmul(invhi,al(i-1,:,:))
        ul2(i-1,:,:)=ui
    enddo
    !sequence for i == j
    ll3=ll1
    do i=1,n
        hi=bl(i,:,:)
        if(i/=1) then
            hi=hi-matmul(ll3(i-1,:,:),ul1(i-1,:,:))
        endif
        if(i/=n) then
            ui=matmul(invhl2(i+1,:,:),al(i,:,:))
            ul3(i,:,:)=ui
            hi=hi-matmul(ll2(i,:,:),ui)
        endif
        hl3(i,:,:)=hi
        invhi=hi
        call zinv(p,invhi)
        invhl3(i,:,:)=invhi
    enddo
end subroutine fget_tlu_seqsn

!get the sequences of twisted LU decomposition for tridiagonal `Scalar` matrix.
!A=LU.
!reference -> http://dx.doi.org/10.1016/j.amc.2005.11.098

!input:
!al/bl/cl
!the lower diagonal/diagonal/upper diagonal part of matrix.

!return:
!(ll1,ll1,ll1,ul2,ul2,ul2,hl3,hl3,hl3,invhl1,invhl2,invhl3), they are sequences defining L,U
subroutine fget_tlu_seqs1(al,bl,cl,ll1,ll2,ll3,ul1,ul2,ul3,hl1,hl2,hl3,invhl1,invhl2,invhl3,n)
    implicit none
    integer,intent(in) :: n
    complex*16,intent(in),dimension(n) :: bl
    complex*16,intent(in),dimension(n-1) :: al,cl
    complex*16,intent(out),dimension(n-1) :: ll1,ll2,ll3,ul1,ul2,ul3
    complex*16,intent(out),dimension(n) :: hl1,hl2,hl3,invhl1,invhl2,invhl3
    complex*16 :: bi,hi,ui,invhi
    integer :: i
    
    !f2py intent(in) :: n,al,bl,cl
    !f2py intent(out) :: ll1,ll2,ll3,ul1,ul2,ul3,hl1,hl2,hl3,invhl1,invhl2,invhl3

    !sequence for i < j
    ll1=al
    do i=1,n-1
        bi=bl(i)
        hi=bi
        if(i/=1) then
            hi=hi-ll1(i-1)*ui
        endif
        invhi=1D0/hi
        hl1(i)=hi
        invhl1(i)=invhi
        ui=invhi*cl(i)
        ul1(i)=ui
    enddo
    !sequence for i > j
    ll2=cl
    do i=n,2,-1
        hi=bl(i)
        if(i/=n) then
            hi=hi-ll2(i)*ui
        endif
        invhi=1D0/hi
        hl2(i)=hi
        invhl2(i)=invhi
        ui=invhi*al(i-1)
        ul2(i-1)=ui
    enddo
    !sequence for i == j
    ll3=ll1
    do i=1,n
        hi=bl(i)
        if(i/=1) then
            hi=hi-ll3(i-1)*ul1(i-1)
        endif
        if(i/=n) then
            ui=invhl2(i+1)*al(i)
            ul3(i)=ui
            hi=hi-ll2(i)*ui
        endif
        hl3(i)=hi
        invhi=1D0/hi
        invhl3(i)=invhi
    enddo
end subroutine fget_tlu_seqs1

!get the sequences of twisted LU decomposition for `Block!` tridiagonal block matrix, the one by one version.
!A=LU.
!reference -> http://dx.doi.org/10.1016/j.amc.2005.11.098

!input
!al/bl/cl:
    !the lower diagonal/diagonal/upper diagonal part of matrix.
!which:
    !the taks to run `1` for i<j, `2` for i>j.

!return:
!(ll,ul,hl,invhl), they are sequences defining L,U
subroutine fget_tlu_seqn(al,bl,cl,ll,ul,hl,invhl,n,p,which)
    implicit none
    integer,intent(in) :: n,p,which
    complex*16,intent(in),dimension(n,p,p) :: bl
    complex*16,intent(in),dimension(n-1,p,p) :: al,cl
    complex*16,intent(out),dimension(n-1,p,p) :: ll,ul
    complex*16,intent(out),dimension(n,p,p) :: hl,invhl
    complex*16,dimension(p,p) :: bi,hi,ui,invhi
    integer :: i
    
    !f2py intent(in) :: n,p,which,al,bl,cl
    !f2py intent(out) :: ll,ul,hl,invhl

    select case(which)
    case(1)
        !sequence for i < j
        ll=al
        do i=1,n
            bi=bl(i,:,:)
            hi=bi
            if(i/=1) then
                hi=hi-matmul(ll(i-1,:,:),ui)
            endif
            invhi=hi
            call zinv(p,invhi)
            hl(i,:,:)=hi
            invhl(i,:,:)=invhi
            if(i/=n) then
                ui=matmul(invhi,cl(i,:,:))
                ul(i,:,:)=ui
            endif
        enddo
    case(2)
        !sequence for i > j
        ll=cl
        do i=n,1,-1
            hi=bl(i,:,:)
            if(i/=n) then
                hi=hi-matmul(ll(i,:,:),ui)
            endif
            invhi=hi
            call zinv(p,invhi)
            hl(i,:,:)=hi
            invhl(i,:,:)=invhi
            if(i/=1) then
                ui=matmul(invhi,al(i-1,:,:))
                ul(i-1,:,:)=ui
            endif
        enddo
    endselect
end subroutine fget_tlu_seqn


!get the sequences of twisted LU decomposition for `Scalar!` tridiagonal block matrix, the one by one version.
!A=LU.
!reference -> http://dx.doi.org/10.1016/j.amc.2005.11.098

!input
!al/bl/cl:
    !the lower diagonal/diagonal/upper diagonal part of matrix.
!which:
    !the taks to run `1` for i<j, `2` for i>j.

!return:
!(ll,ul,hl), they are sequences defining L,U
subroutine fget_tlu_seq1(al,bl,cl,ll,ul,hl,n,which)
    implicit none
    integer,intent(in) :: n,which
    complex*16,intent(in),dimension(n) :: bl
    complex*16,intent(in),dimension(n-1) :: al,cl
    complex*16,intent(out),dimension(n-1) :: ll,ul
    complex*16,intent(out),dimension(n) :: hl
    complex*16 :: bi,hi,ui
    integer :: i
    
    !f2py intent(in) :: n,p,which,al,bl,cl
    !f2py intent(out) :: ll,ul,hl

    select case(which)
    case(1)
        !sequence for i < j
        ll=al
        do i=1,n
            bi=bl(i)
            hi=bi
            if(i/=1) then
                hi=hi-ll(i-1)*ui
            endif
            hl(i)=hi
            if(i/=n) then
                ui=cl(i)/hi
                ul(i)=ui
            endif
        enddo
    case(2)
        !sequence for i > j
        ll=cl
        do i=n,1,-1
            hi=bl(i)
            if(i/=n) then
                hi=hi-ll(i)*ui
            endif
            hl(i)=hi
            if(i/=1) then
                ui=al(i-1)/hi
                ul(i-1)=ui
            endif
        enddo
    endselect
end subroutine fget_tlu_seq1

!get the diagonal part of UDL decomposition.

!input
!al/bl/cl:
!    the lower,diagonal,upper part of tridiagonal matrix
!udl:
!    decompose as UDL if True else LDU

!output
!dl:
!    the diagonal part of ldu/udl decomposition
subroutine fget_dl(al,bl,cl,dl,udl,n)
    implicit none
    integer,intent(in) :: n
    complex*16,intent(in),dimension(n) :: bl
    complex*16,intent(in),dimension(n-1) :: al,cl
    complex*16,intent(out),dimension(n) :: dl
    complex*16,dimension(n-1) :: ul
    complex*16 :: di
    integer :: i
    logical :: udl
    
    !f2py intent(in) :: n,al,bl,cl,udl
    !f2py intent(out) :: dl

    ul=al*cl
    if(udl) then
        di=bl(n)
        dl(n)=di
        do i=n-1,1,-1
            di=bl(i)-ul(i)/di
            dl(i)=di
        enddo
    else
        di=bl(1)
        dl(1)=di
        do i=2,n
            di=bl(i)-ul(i-1)/di
            dl(i)=di
        enddo
    endif
end subroutine fget_dl

!get the diagonal part of UDL decomposition, the Block version.

!input
!al/bl/cl:
!    the lower,diagonal,upper part of tridiagonal matrix
!udl:
!    decompose as UDL if True else LDU

!output
!dl:
!    the diagonal part of ldu/udl decomposition
subroutine fget_dln(al,bl,cl,dl,invdl,udl,n,p)
    implicit none
    integer,intent(in) :: n,p
    complex*16,intent(in),dimension(n,p,p) :: bl
    complex*16,intent(in),dimension(n-1,p,p) :: al,cl
    complex*16,intent(out),dimension(n,p,p) :: dl,invdl
    complex*16,dimension(p,p) :: di,invdi
    integer :: i
    logical :: udl
    
    !f2py intent(in) :: n,p,al,bl,cl,udl
    !f2py intent(out) :: dl,invdl

    if(udl) then
        do i=n,1,-1
            di=bl(i,:,:)
            if(i/=n) then
                di=di-matmul(cl(i,:,:),matmul(invdi,al(i,:,:)))
            endif
            dl(i,:,:)=di
            invdi=di
            call zinv(p,invdi)
            invdl(i,:,:)=invdi
        enddo
    else
        do i=1,n
            di=bl(i,:,:)
            if(i/=1) then
                di=di-matmul(al(i-1,:,:),matmul(invdi,cl(i-1,:,:)))
            endif
            dl(i,:,:)=di
            invdi=di
            call zinv(p,invdi)
            invdl(i,:,:)=invdi
        enddo
    endif
end subroutine fget_dln


!get u,v vectors defining inversion,
!the inversion of A is:
!    inv(A) = [u1v1/2, u1v2 ... u1vn] + h.c.
!             [    , u2v2/2 ... u2vn]
!             [         ...         ]
!             [         ...   unvn/2]
!
!input
!invdu/invdl:
!    the inverse of diagonal part of UDL and LDU decomposition.
!cl:
!    the upper part of tridiagonal matrix.
!
!output
!ul,vl:
!    the u,v vectors defining inversion of a matrix.
subroutine fget_uv(invdu,invdl,cl,ul,vl,n)
    implicit none
    integer,intent(in) :: n
    complex*16,intent(in),dimension(n) :: invdu,invdl
    complex*16,intent(in),dimension(n-1) :: cl
    complex*16,intent(out),dimension(n) :: ul,vl
    complex*16 :: ui,vi
    integer :: i
    
    !f2py intent(in) :: n,invdu,invdl,cl
    !f2py intent(out) :: ul,vl

    !get vl
    vi=1D0*invdu(1)
    vl(1)=vi
    do i=2,n
        vi=-vi*cl(i-1)*invdu(i)
        vl(i)=vi
    enddo
    !get ul
    ui=invdl(n)/vl(n)
    ul(n)=ui
    do i=n-1,1,-1
        ui=-ui*cl(i)*invdl(i)
        ul(i)=ui
    enddo
end subroutine fget_uv


!get u,v vectors defining inversion, the Block version
!the inversion of A is:
!    inv(A) = [u1v1/2, u1v2 ... u1vn] + h.c.
!             [    , u2v2/2 ... u2vn]
!             [         ...         ]
!             [         ...   unvn/2]
!
!input
!invdu/invdl:
!    the inverse of diagonal part of UDL and LDU decomposition.
!cl:
!    the upper part of tridiagonal matrix.
!
!output
!ul,vl:
!    the u,v vectors defining inversion of a matrix.
subroutine fget_uvn(invdu,invdl,cl,ul,vl,n,p)
    implicit none
    integer,intent(in) :: n,p
    complex*16,intent(in),dimension(n,p,p) :: invdu,invdl
    complex*16,intent(in),dimension(n-1,p,p) :: cl
    complex*16,intent(out),dimension(n,p,p) :: ul,vl
    complex*16,dimension(p,p) :: ui,vi
    integer :: i
    
    !f2py intent(in) :: n,p,invdu,invdl,cl
    !f2py intent(out) :: ul,vl

    !get vl
    vi=invdu(1,:,:)
    vl(1,:,:)=vi
    do i=1,n
        vi=matmul(matmul(vi,invdu(i,:,:)),cl(i-1,:,:))
        vl(i,:,:)=vi
    enddo
    !get ul
    call zinv(p,vi)
    ui=matmul(invdl(n-1,:,:),vi)
    ul(n,:,:)=ui
    do i=n-1,1,-1
        ui=matmul(cl(i,:,:),matmul(invdl(i,:,:),ui))
        ul(i,:,:)=ui
    enddo
end subroutine fget_uvn

