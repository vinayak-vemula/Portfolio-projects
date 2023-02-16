select *
from PortfolioProject.dbo.Nashville_housing

-- Standardize Date Format

select saledate from PortfolioProject.dbo.Nashville_housing

select SaleDate, convert(Date,saledate)
from nashville_housing


update nashville_housing 
set SaleDate= convert(date,saledate)

select saledate from PortfolioProject.dbo.Nashville_housing

--Populate Property Address Data

select * 
from PortfolioProject.dbo.Nashville_housing
where propertyaddress is  null

Select a.ParcelID, a.PropertyAddress, b.ParcelID, b.PropertyAddress,
   ISNULL(a.PropertyAddress,b.PropertyAddress)
From PortfolioProject.dbo.Nashville_housing a
JOIN PortfolioProject.dbo.Nashville_housing b
	on a.ParcelID = b.ParcelID
	AND a.[UniqueID ] <> b.[UniqueID ]
Where a.PropertyAddress is null


update a
SET PropertyAddress = ISNULL(a.PropertyAddress,b.PropertyAddress)
From PortfolioProject.dbo.Nashville_housing a
JOIN PortfolioProject.dbo.Nashville_housing b
	on a.ParcelID = b.ParcelID
	AND a.[UniqueID ] <> b.[UniqueID ]
Where a.PropertyAddress is null


-- Breaking out Address into Individual Columns (Address, City, State)

select propertyaddress from PortfolioProject.dbo.Nashville_housing

select
substring(propertyaddress,1, charindex (',',propertyaddress) -1) as Address
,substring(propertyaddress, charindex(',',propertyaddress) +1 , LEN(PropertyAddress)) as Address
from PortfolioProject.dbo.Nashville_housing

alter table nashville_housing
add property_address nvarchar(120)

update [Nashville_Housing ]
set property_address = substring(propertyaddress,1, charindex (',',propertyaddress) -1)

alter table nashville_housing
add property_city nvarchar(120)

update nashville_housing
set property_city =substring(propertyaddress, charindex(',',propertyaddress) +1 , LEN(PropertyAddress))

select * from [Nashville_Housing ]

-- populate owner_address

select OwnerAddress from [Nashville_Housing ]

select 
PARSENAME(REPLACE(owneraddress,',','.'),3),
PARSENAME(REPLACE(owneraddress,',','.'),2),
PARSENAME(REPLACE(owneraddress,',','.'),1)
from [Nashville_Housing ]


alter table nashville_housing
add owner_address nvarchar(120)

update [Nashville_Housing ]
set owner_address = PARSENAME(REPLACE(owneraddress,',','.'),3)


alter table nashville_housing
add ownercity nvarchar(120)

update [Nashville_Housing ]
set ownercity = PARSENAME(REPLACE(owneraddress,',','.'),2)

alter table nashville_housing
add ownerstate nvarchar(120)

update [Nashville_Housing ]
set ownerstate = PARSENAME(REPLACE(owneraddress,',','.'),1)


select * from [Nashville_Housing ]

-- Remove duplicates


WITH RowNumCTE 
AS(
Select *,
	ROW_NUMBER() OVER (
	PARTITION BY ParcelID,
				 PropertyAddress,
				 SalePrice,
				 SaleDate,
				 LegalReference
				 ORDER BY
					UniqueId) row_num

From Nashville_Housing
)
delete 
from RowNumCTE
where row_num > 1;

select * from [Nashville_Housing ]

-- Delete unused column


Select *
From Nashville_Housing

alter table nashville_housing
drop column OwnerAddress, TaxDistrict, PropertyAddress, SaleDate;


select* from [Nashville_Housing ];


