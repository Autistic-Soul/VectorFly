DATA	SEGMENT
	ORG	100H
	VAL	DW	345BH
	DATA_BYTE	DB	12,	3AH
	DATA_WORD	DW	21,	$ + 5,	-5
	DATA_DW	DD	3 * 8,	04030201H
	MESSAGE	DW	'AB'
	DATA1	DB	1,	2H
	EXPR	DW	1,	2
	STR	DB	'WELCOME!'
	S1	DW	'AB'
	S2	DD	'AB'
	OFFAB	DW	S1
DATA	ENDS