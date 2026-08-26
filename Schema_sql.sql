create database clinic_noshow;
use clinic_noshow;

create table patients (
    patient_id bigint primary key,
    age int,
    sexe enum('M', 'F'),
    quartier varchar(100),
    hypertension boolean,
    diabete boolean, 
    alcoolisme boolean,
    handicap int);
    
create table rendez_vous (
    appointment_id bigint primary key,
    ref_patient bigint,
    date_prise_rdv datetime,
    date_consultation date,
    jour_semaine varchar(10),
    sms_recu boolean,
    presence boolean, -- variable cible : 1 = présent, 0 = absent
    foreign key (ref_patient) references patients(patient_id));
    
    -- Taux d'absence par quartier
select p.quartier,
	 count(*) as total_rdv,
     sum(case when r.presence = 0 then 1 else 0 end) as absences,
     round(sum(case when r.presence = 0 then 1 else 0 end)/count(*)*100,2) as taux_absence
from rendez_vous r
join patients p on r.ref_patient = p.patient_id
group by p.quartier
order by taux_absence desc;

-- Délai entre prise de RDV et consultation, par tranche d'age
select 
	case 
      when p.age < 18 then '0-17'
      when p.age < 40 then '18-39'
      when p.age < 65 then '40-64'
    else '65+'
    end as tranche_age,
    avg(datediff(r.date_consultation, r.date_prise_rdv)) as delai_moyen_jours,
    avg(r.presence) as taux_presence
from rendez_vous r
join patients p on r.ref_patient = p.patient_id
group by tranche_age;

create index idx_rdv_patient on rendez_vous(ref_patient);
create index idx_rdv_presence on rendez_vous(presence);