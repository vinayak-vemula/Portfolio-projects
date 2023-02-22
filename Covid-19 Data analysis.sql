-- Covid 19 Data Analysis

select * from 
portfolioproject_3.dbo.coviddeaths

select * from 
portfolioproject_3.dbo.CovidVaccinations

-- Select Data that we are going to be starting with

select location, date, total_cases , new_cases, total_deaths, population
from portfolioproject_3.dbo.coviddeaths

-- how many total deaths vs total cases and percent of deaths

Select Location, date, total_cases,total_deaths, (total_deaths/total_cases)*100 as DeathPercentage
From portfolioproject_3.dbo.coviddeaths

-- Total population and percent of population infected with covid

Select Location, date,population ,total_cases, (total_cases/population)*100 as covidpercent
From portfolioproject_3.dbo.coviddeaths

-- Countries with highest cases 

Select Location, Population, MAX(total_cases) as HighestInfectionCount,  Max((total_cases/population))*100 as Infectedpopulationpercent
From portfolioproject_3.dbo.coviddeaths
group by location, population
order by Infectedpopulationpercent desc

-- Countries with highest death toll

Select Location, Population, MAX(total_deaths) as HighestDeathCount,  Max((total_deaths/population))*100 as deathspopulationpercent
From portfolioproject_3.dbo.coviddeaths
group by location, population
order by deathspopulationpercent desc

--Explore data with respect to continents

-- Highest death tolls wrt continents

Select continent, MAX(total_deaths) as HighestDeathCount
From portfolioproject_3.dbo.coviddeaths
where continent is not null 
group by continent
order by HighestDeathCount desc

-- Overall cases, deaths

select SUM(new_cases) as total_cases, SUM(cast(new_deaths as int)) as total_deaths, 
SUM(cast(new_deaths as int))/SUM(New_Cases)*100 as DeathPercentage
From portfolioproject_3.dbo.coviddeaths
where continent is not null

-- Checking number of population that has received vaccination by joining vaccination table


With PopvsVac (Continent, Location, Date, Population, New_Vaccinations, RollingPeopleVaccinated)
as
(Select d.continent, d.location, d.date, d.population, v.new_vaccinations
, SUM(CONVERT(int,v.new_vaccinations)) OVER (Partition by d.Location Order by d.location, d.Date) as RollingPeopleVaccinated
 from portfolioproject_3.dbo.coviddeaths d
 join portfolioproject_3.dbo.CovidVaccinations v
 On d.location = v.location
	and d.date = v.date
where d.continent is not null 
)
Select *, (RollingPeopleVaccinated/Population)*100 as vaccinationpercent
From PopvsVac

-- create view for total percent of vaccination

DROP Table if exists #populationvaccinationpercent

create view populationvaccinationpercent as
Select d.continent, d.location, d.date, d.population, v.new_vaccinations
, SUM(CONVERT(int,v.new_vaccinations)) OVER (Partition by d.Location Order by d.location, d.Date) as RollingPeopleVaccinated
 from portfolioproject_3.dbo.coviddeaths d
 join portfolioproject_3.dbo.CovidVaccinations v
 On d.location = v.location
	and d.date = v.date
where d.continent is not null 




