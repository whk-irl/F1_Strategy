"use strict";
// The only browser/server exchange is Streamlit's component protocol. No API keys
// or provider requests cross this boundary; animation stays in this iframe.
const $ = id => document.getElementById(id);
const state = {payload:null, race:null, t:0, playing:!matchMedia('(prefers-reduced-motion: reduce)').matches,
  speed:1, driver:1, labels:true, follow:false, lastFrame:0, lastPanel:0, lastSave:0, rows:new Map(), planKey:null};
const tyreColors = {SOFT:'#f55d62',MEDIUM:'#f0d35a',HARD:'#e4e5e9',INTERMEDIATE:'#63d28d',WET:'#65a3eb',UNKNOWN:'#737d8c'};
const post = (type, body={}) => parent.postMessage({isStreamlitMessage:true,type,...body}, '*');
function height(){post('streamlit:setFrameHeight',{height:Math.ceil(document.documentElement.getBoundingClientRect().height)});}
function send(action,value){save();post('streamlit:setComponentValue',{value:{action,value,nonce:Date.now()+Math.random()},dataType:'json'});}
function txt(id,value){const el=$(id); const next=String(value); if(el.textContent!==next)el.textContent=next;}
function fixed(n,d=1){return n==null||!Number.isFinite(Number(n))?'—':Number(n).toFixed(d);}
function numericGap(value){if(value==null||(typeof value==='string'&&!value.trim())||typeof value==='boolean')return null;const n=Number(value);return Number.isFinite(n)?n:null;}
function clock(t,full=false){t=Math.max(0,Math.floor(t));let s=String(t%60).padStart(2,'0'),m=String(Math.floor(t/60)%60).padStart(2,'0');return full?`${String(Math.floor(t/3600)).padStart(2,'0')}:${m}:${s}`:`${Math.floor(t/60)}:${s}`;}
function lapTime(t){return t==null?'—':`${Math.floor(t/60)}:${(t%60).toFixed(3).padStart(6,'0')}`;}
function gap(v,leader=false){return leader?'LEADER':v==null?'—':typeof v==='number'?`+${v.toFixed(3)}`:String(v);}
function latest(rows,t,column=0){let lo=0,hi=(rows||[]).length;while(lo<hi){const mid=(lo+hi)>>>1;if(rows[mid][column]<=t)lo=mid+1;else hi=mid;}return lo?rows[lo-1]:null;}
function latestEvent(rows,t){let lo=0,hi=(rows||[]).length;while(lo<hi){const mid=(lo+hi)>>>1;if(rows[mid].t<=t)lo=mid+1;else hi=mid;}return lo?rows[lo-1]:null;}
function save(){if(!state.race)return;try{sessionStorage.setItem('pitwall-preferences',JSON.stringify({speed:state.speed,labels:state.labels,follow:state.follow}));}catch(_){}}
function setOptions(id,values,selected){const el=$(id);el.replaceChildren(...values.map(([v,label])=>{const op=document.createElement('option');op.value=v;op.textContent=label;return op;}));el.value=selected;}
function render(payload){
  state.payload=payload;
  setOptions('season',payload.years.map(y=>[y,y]),payload.year);
  setOptions('session',payload.catalog.map(s=>[s.session_key,`${s.circuit_short_name} · ${s.country_name}`]),payload.race?.id||payload.catalog.at(-1)?.session_key);
  $('replay-mode').classList.toggle('active',payload.mode==='replay');$('live-mode').classList.toggle('active',payload.mode==='live');
  $('season').disabled=payload.mode==='live';$('session').disabled=payload.mode==='live';
  $('notice').hidden=!payload.notice;$('race-content').hidden=!!payload.notice||!payload.race;
  txt('feed-label',payload.mode==='live'?'LIVE CONNECTION':'RECORDED TELEMETRY');
  if(payload.notice){state.race=null;txt('notice-title',payload.notice.title);txt('notice-detail',payload.notice.detail);txt('event-title',payload.weekend?'United States Grand Prix':'Live race center');txt('event-subtitle',payload.weekend?'Austin / Weekend preparation and local race test':'Ready when the lights go out.');txt('round-title',`RACE PREPARATION / ${payload.year}`);height();return;}
  const race=payload.race;if(!race)return;
  if(state.race?.id!==race.id){
    state.t=0;state.planKey=null;state.playing=!matchMedia('(prefers-reduced-motion: reduce)').matches;
    try{const saved=JSON.parse(sessionStorage.getItem('pitwall-preferences'));if(saved){state.speed=saved.speed||1;state.labels=saved.labels!==false;state.follow=!!saved.follow;}}catch(_){}
  }
  state.race=race;state.driver=payload.driver;
  if(!race.drivers.some(d=>d.driver_number===state.driver))state.driver=race.drivers[0]?.driver_number;
  if(payload.mode==='live'){state.t=Math.max(0,race.duration-(race.capabilities?.position===false?0:20));state.playing=false;state.snapshotAt=performance.now();}
  state.t=Math.max(0,Math.min(state.t,race.duration));
  const m=race.meta;
  const eventNames={Monza:'Italian',Melbourne:'Australian',Shanghai:'Chinese',Suzuka:'Japanese',Montreal:'Canadian','Monte Carlo':'Monaco',Catalunya:'Spanish',Spielberg:'Austrian',Silverstone:'British','Spa-Francorchamps':'Belgian',Hungaroring:'Hungarian',Zandvoort:'Dutch',Baku:'Azerbaijan',Singapore:'Singapore'};
  txt('event-title',`${eventNames[m.circuit_short_name]||m.location} Grand Prix`);txt('event-subtitle',`${m.location}  /  ${new Date(m.date_start).toLocaleDateString('en-GB',{day:'2-digit',month:'long',year:'numeric',timeZone:'UTC'})}  /  ${m.session_name}`);
  txt('round-title',`FORMULA 1 / ${m.year} SEASON`);txt('total-laps',payload.mode==='live'&&!race.distance_confirmed?'/ —':`/ ${race.total_laps}`);txt('map-name',m.circuit_short_name.toUpperCase());txt('map-location',m.circuit_short_name.toUpperCase());
  txt('map-source',payload.mode==='live'?(race.capabilities?.position===false?'STATIC CIRCUIT · LIVE GPS UNAVAILABLE':'DELAYED GPS · 15s REFRESH'):'RECORDED GPS · INTERPOLATED');
  $('track-canvas').setAttribute('aria-label',race.capabilities?.position===false?'Static circuit map; live car positions unavailable':'Circuit map with recorded car positions');
  document.querySelector('.strategy-badge').textContent=payload.mode==='live'?'SHADOW TEST':'RACE MODEL';
  document.querySelector('.strategy-footnote').textContent=race.capabilities?.position===false?`Experimental local shadow test. Historical measured pit loss and practice-informed tyres guide the nominal plan. Live GPS, car telemetry and measured pit-lane times are unavailable. Rejoin uses current gaps; future rival traffic is unknown.${race.capture_ok===false?' Local capture failed; check server storage.':''}`:'Recommendation combines measured pit loss, tyre age, race state, nearby gaps, rival tyres and recent stops. Rejoin outcomes are ranges, not guarantees. The dry-race dynamic program is a research reference.';
  document.querySelector('footer span:nth-child(2)').textContent=race.capabilities?.position===false?'Data: public F1 timing via FastF1 · Experimental local test · Unofficial':'Data: OpenF1 · Independent research project · Unofficial';
  txt('timeline-note',race.capabilities?.position===false?'Timing samples · no live GPS':'Actual positions · smooth playback');
  txt('playback-label',payload.mode==='live'?(race.capabilities?.position===false?'LIVE TIMING · EXPERIMENTAL':'LIVE TIMING · DELAYED'):'RACE REPLAY');
  for(const id of ['play','restart','seek','playback-speed'])$(id).disabled=payload.mode==='live';
  $('playback-speed').value=state.speed;$('labels').classList.toggle('active',state.labels);$('labels').setAttribute('aria-pressed',String(state.labels));$('focus').classList.toggle('active',state.follow);$('focus').setAttribute('aria-pressed',String(state.follow));
  buildTower();fitTrack();updatePanels();height();
}
function driverInfo(number){return state.race.drivers.find(d=>d.driver_number===number)||{driver_number:number,name_acronym:String(number),full_name:'Unknown driver',team_colour:'73839b',team_name:'—'};}
function currentLap(number){return latest(state.race.laps[number],state.t);}
function completedLap(number){const observed=state.race.latest_completed?.[number];if(observed&&observed[1]<=state.t)return observed;const rows=state.race.laps[number]||[];for(let i=rows.length-1;i>=0;i--)if(rows[i][1]!=null&&rows[i][1]<=state.t)return rows[i];return null;}
function currentStint(number,lap){return state.race.stints.filter(s=>s.driver_number===number&&s.lap_start<=lap).sort((a,b)=>Number(!!a.current_sample)-Number(!!b.current_sample)||a.stint_number-b.stint_number).at(-1);}
function isOut(number){if(state.payload.mode==='live'&&!state.race.distance_confirmed)return false;const laps=state.race.laps[number]||[];const last=laps.at(-1);return last&&last[2]<state.race.total_laps-2&&state.t>(last[1]??last[0]+120)+180&&state.trackFlag!=='RED FLAG';}
function buildTower(){
  const tower=$('timing-tower');tower.replaceChildren();state.rows.clear();
  for(const d of state.race.drivers){
    const row=document.createElement('button');row.className='timing-row';row.dataset.driver=d.driver_number;row.style.setProperty('--team',`#${d.team_colour||'73839b'}`);row.setAttribute('aria-label',`Follow ${d.full_name}`);
    row.innerHTML='<span class="position"></span><span class="timing-name"><i class="team-line"></i><span></span></span><span class="tyre unknown">?</span><span class="timing-gap"></span>';
    row.querySelector('.timing-name span').textContent=d.name_acronym||d.driver_number;
    row.onclick=()=>{if(state.driver===d.driver_number)return;state.driver=d.driver_number;state.planKey=null;updatePanels();drawTrack();send('driver',d.driver_number);};
    state.rows.set(d.driver_number,row);tower.append(row);
  }
}
function flagState(){
  const messages=state.race.messages.filter(m=>m.t<=state.t);
  let flag='TRACK CLEAR',style='',neutral=false;const yellow=new Set();
  for(const m of messages){const s=(m.message||'').toUpperCase();
    const sector=s.match(/TRACK SECTOR (\d+)/)?.[1]||'TRACK';
    if(['YELLOW','DOUBLE YELLOW'].includes(m.flag))yellow.add(sector);
    if(m.flag==='CLEAR')yellow.delete(sector);
    if(s.includes('RED FLAG')||m.flag==='RED'){flag='RED FLAG';style='red';neutral=true;}
    if(s.includes('SAFETY CAR DEPLOYED')&&!s.includes('VIRTUAL')){flag='SAFETY CAR';style='yellow';neutral=true;}
    if(s.includes('VIRTUAL SAFETY CAR DEPLOYED')){flag='VIRTUAL SAFETY CAR';style='yellow';neutral=true;}
    if(s.includes('SAFETY CAR IN THIS LAP')){flag='SC IN THIS LAP';style='yellow';neutral=true;}
    if(s.includes('VIRTUAL SAFETY CAR ENDING')||s.includes('SAFETY CAR IN THE PIT LANE')||m.flag==='GREEN'||s.includes('GREEN LIGHT')){flag='TRACK CLEAR';style='';neutral=false;}
    if(s==='SESSION STARTED'||s==='RACE START'||s==='STANDING START'){flag='TRACK CLEAR';style='';neutral=false;yellow.clear();}
    if(m.flag==='CHEQUERED'){flag='CHEQUERED FLAG';style='';}
  }
  if(flag==='TRACK CLEAR'&&yellow.size){flag='YELLOW FLAG';style='yellow';}
  if(state.race.track_status&&(!messages.length||(state.race.track_status_t??-1)>=messages.at(-1).t)){const status=String(state.race.track_status);if(status==='4'){flag='SAFETY CAR';style='yellow';neutral=true;}else if(['6','7'].includes(status)){flag='VIRTUAL SAFETY CAR';style='yellow';neutral=true;}else if(status==='5'){flag='RED FLAG';style='red';neutral=true;}else if(status==='1'){flag='TRACK CLEAR';style='';neutral=false;}}
  return {flag,style,neutral};
}
function updatePanels(){
  const r=state.race;if(!r)return;
  const flag=flagState();state.trackFlag=flag.flag;
  const order=r.drivers.map(d=>({d,p:latest(r.positions[d.driver_number],state.t)?.[1]??99})).sort((a,b)=>a.p-b.p||a.d.driver_number-b.d.driver_number);
  let leaderLap=1;
  for(const {d,p} of order){
    const n=d.driver_number,row=state.rows.get(n),lap=currentLap(n),stint=currentStint(n,lap?.[2]||1),int=latest(r.intervals[n],state.t);
    row.style.order=p;row.classList.toggle('selected',n===state.driver);row.setAttribute('aria-pressed',String(n===state.driver));
    row.querySelector('.position').textContent=p<99?p:'—';row.querySelector('.position').classList.toggle('top',p<=3);
    const compound=stint?.compound||'UNKNOWN',tyre=row.querySelector('.tyre');tyre.className=`tyre ${compound.toLowerCase()}`;tyre.textContent=compound==='UNKNOWN'?'?':compound[0];
    const g=row.querySelector('.timing-gap'),v=int&&state.t-int[0]<45?int[1]:null;g.textContent=isOut(n)?'NO DATA':gap(v,p===1);g.classList.toggle('leader',p===1||isOut(n));
    $('timing-tower').append(row);if(p===1)leaderLap=lap?.[2]||1;
  }
  txt('leader-lap',leaderLap);$('lap-progress').style.width=state.payload.mode==='live'&&!r.distance_confirmed?'0%':`${leaderLap/r.total_laps*100}%`;txt('race-clock',clock(state.t,true));
  txt('track-status',flag.flag);$('track-status').className=`status-chip ${flag.style}`;
  const weatherSample=latestEvent(r.weather,state.t),weather=state.payload.mode==='live'&&weatherSample&&state.t-weatherSample.t>600?null:weatherSample;txt('air-temp',weather?`${fixed(weather.air_temperature)}°`:'—');txt('track-temp',weather?`${fixed(weather.track_temperature)}°`:'—');txt('wind',weather?`${fixed(weather.wind_speed)} m/s`:'—');
  const n=state.driver,d=driverInfo(n),lap=currentLap(n),done=completedLap(n),stint=currentStint(n,lap?.[2]||1),compound=stint?.compound||'UNKNOWN';
  const age=stint&&!(stint.current_sample&&stint.tyre_age_at_start==null)?Math.max(0,(lap?.[2]||1)-stint.lap_start+(stint.tyre_age_at_start||0)):null;
  const p=latest(r.positions[n],state.t)?.[1];const interval=latest(r.intervals[n],state.t);
  txt('telemetry-driver',`${d.name_acronym} / ${n}`);$('telemetry-driver').style.setProperty('--team',`#${d.team_colour||'73839b'}`);
  txt('driver-number',n);txt('driver-name',d.full_name);txt('driver-team',d.team_name);txt('driver-position',p?`P${p}`:'P—');document.querySelector('.driver-panel').style.setProperty('--team',`#${d.team_colour||'73839b'}`);
  const ownPits=r.pits.filter(s=>s.driver_number===n&&s.t<=state.t),lastPit=ownPits.at(-1);
  const lane=lastPit?.lane_duration,pitDisplay=lane==null?'—':lane>=60?`${Math.floor(lane/60)}m${fixed(lane%60)}s`:`${fixed(lane)}s`;
  const intervalAge=r.interval_ages?.[n],intervalFresh=r.interval_ages==null||(intervalAge!=null&&intervalAge+(performance.now()-state.snapshotAt)/1000<45);
  txt('driver-interval',interval&&state.t-interval[0]<45&&intervalFresh?gap(interval[2],p===1):'—');txt('driver-stops',r.pit_counts?.[n]??ownPits.length);txt('driver-last-pit',pitDisplay);txt('driver-lap',lap?.[2]||'—');
  $('driver-last-pit').title=lane==null?'No completed stop recorded':`Recorded lane time ${fixed(lane,3)} s${lastPit.stop_duration==null?'':` · stationary service ${fixed(lastPit.stop_duration,3)} s`}${lane>45?' · extended stop, excluded from routine forecast':''}`;
  $('tyre-badge').className=`tyre ${compound.toLowerCase()}`;txt('tyre-badge',compound==='UNKNOWN'?'?':compound[0]);txt('tyre-name',compound);txt('tyre-age',age==null?'AGE UNAVAILABLE':`${age} LAPS ON TYRE`);txt('stint-label',`STINT ${stint?.stint_number||'—'}`);
  const history=r.stints.filter(s=>s.driver_number===n&&s.lap_start<=(lap?.[2]||1));
  $('stint-history').replaceChildren(...history.filter(s=>!s.current_sample&&s.lap_start>=1).map(s=>{const el=document.createElement('span');el.style.setProperty('--tyre',tyreColors[s.compound]||tyreColors.UNKNOWN);el.style.width=`${(Math.min(s.lap_end||lap?.[2]||1,lap?.[2]||1)-s.lap_start+1)/r.total_laps*100}%`;el.title=`${s.compound} · lap ${s.lap_start} onwards`;return el;}));
  txt('last-lap',lapTime(done?.[3]));for(let i=1;i<=3;i++)txt(`sector-${i}`,done?.[3+i]==null?'—':fixed(done[3+i],3));
  const msgs=r.messages.filter(m=>m.t<=state.t).slice(-3).reverse();
  const messageKey=msgs.map(m=>m.t+m.message).join('|');
  if($('race-messages').dataset.key!==messageKey){$('race-messages').dataset.key=messageKey;$('race-messages').replaceChildren(...msgs.map(m=>{const el=document.createElement('div');el.className=`race-message ${m.flag==='RED'?'red':m.flag==='YELLOW'||(m.message||'').includes('SAFETY CAR')?'yellow':''}`;const time=document.createElement('time');time.textContent=clock(m.t,true)+' / '+(m.category||'RACE CONTROL').toUpperCase();const text=document.createElement('span');text.textContent=m.message;el.append(time,text);return el;}));if(!msgs.length){const p=document.createElement('p');p.className='empty-message';p.textContent='No race-control messages at this point in the race.';$('race-messages').append(p);}height();}
  txt('follow-caption',`FOLLOWING ${d.name_acronym} · #${n}${isOut(n)?' · NO LAP DATA':''}`);
  const gps=latest(r.locations[n],state.t);txt('gps-health',r.capabilities?.position===false?'TIMING ONLY · NO LIVE GPS':gps&&state.t-gps[0]<10?'GPS POSITION FEED':'POSITION UNAVAILABLE');
  updateTelemetry();updateStrategy(compound,age,lap?.[2]||1,history,flag,weather);
  txt('playback-time',`${clock(state.t)} / ${clock(r.duration)}`);$('seek').value=state.t/r.duration*1000;$('seek').style.setProperty('--progress',`${state.t/r.duration*100}%`);
  txt('play',state.playing?'Ⅱ':'▶');$('play').setAttribute('aria-label',state.playing?'Pause replay':'Play replay');
  const liveAge=state.payload.freshness==null?null:state.payload.freshness+(performance.now()-state.snapshotAt)/1000;
  const liveDelay=liveAge==null?null:Math.max(liveAge,r.duration-state.t+(performance.now()-state.snapshotAt)/1000);
  txt('playback-status',state.payload.mode==='live'?(liveAge==null?'NO SIGNAL':liveAge<45?(r.capabilities?.position===false?'TIMING ONLY':'BUFFERED'):'STALE'):state.playing?'PLAYING':'PAUSED');
  txt('feed-label',state.payload.mode==='live'?(liveDelay==null?'LIVE · NO SIGNAL':`${r.capabilities?.position===false?'EXPERIMENTAL TIMING':'LIVE'} · ${fixed(liveDelay,0)}s DELAY`):'RECORDED TELEMETRY');
}
function updateTelemetry(){
  const rows=state.payload.driver===state.driver?state.payload.telemetry:[],sample=latest(rows,state.t);
  const valid=sample&&state.t-sample[0]<3&&!isOut(state.driver);const v=valid?sample:null;
  txt('speed',v?.[1]??'—');txt('gear',v?(v[4]===0?'N':v[4]??'—'):'—');txt('rpm',v?.[6]==null?'— RPM':`${Number(v[6]).toLocaleString('en-US')} RPM`);
  txt('throttle-value',v?.[2]==null?'—':`${v[2]}%`);txt('brake-value',v?.[3]==null?'—':`${v[3]}%`);
  $('throttle-bar').style.width=`${v?.[2]||0}%`;$('brake-bar').style.width=`${v?.[3]||0}%`;
  const drs=v?.[5],open=[10,12,14].includes(drs);txt('drs',drs==null?'DRS —':open?'DRS OPEN':drs===8?'DRS AVAILABLE':[0,1].includes(drs)?'DRS CLOSED':'DRS UNKNOWN');$('drs').classList.toggle('open',open);
  const trace=valid?rows.filter(p=>p[0]>=state.t-16&&p[0]<=state.t&&p[1]!=null):[];
  $('speed-trace').setAttribute('d',trace.map((p,i)=>`${i?'L':'M'}${((p[0]-state.t+16)/16*230).toFixed(1)},${(25-p[1]/380*24).toFixed(1)}`).join(' '));
}
function nominalPlan(remaining,compound,age,mask,pitLoss){
  // Finite-horizon lap-cost DP. A stop serves the current tyres this lap and
  // starts the chosen compound next lap. Two dry compounds are required.
  const calibration=state.payload.weekend?.calibration;
  const rates=[.08,.05,.03].map((prior,i)=>{const value=Number(calibration?.degrade_rate?.[i]);return Number.isFinite(value)&&value>0&&value<.5?value:prior;}),cliffs=[22,32,42].map((prior,i)=>{const value=Number(calibration?.cliff_lap?.[i]);return Number.isFinite(value)&&value>=5&&value<=100?value:prior;}),offset=[0,.25,.55],memo=new Map();
  function solve(left,c,a,used){
    if(left===0)return {cost:(used&(used-1))===0?1e5:0,actions:[]};
    const key=`${left},${c},${a},${used}`;if(memo.has(key))return memo.get(key);
    const lapCost=offset[c]+rates[c]*a+Math.max(0,a-cliffs[c])*rates[c]*3;
    let tail=solve(left-1,c,a+1,used),best={cost:lapCost+tail.cost,actions:[-1,...tail.actions]};
    if(left>1)for(let next=0;next<3;next++){tail=solve(left-1,next,0,used|(1<<next));const cost=lapCost+pitLoss+tail.cost;if(cost<best.cost)best={cost,actions:[next,...tail.actions]};}
    memo.set(key,best);return best;
  }
  return solve(remaining,compound,age,mask).actions;
}
function pitForecast(flag){
  const base=state.payload.pit_time;if(!base)return null;
  const measured=state.race.pits.filter(p=>p.t<=state.t&&p.lane_duration>=12&&p.lane_duration<=45).map(p=>Number(p.lane_duration)).sort((a,b)=>a-b);
  const median=measured.length?measured[Math.floor(measured.length/2)]:base.lane_s;
  const weight=measured.length/(measured.length+8),shift=weight*(median-base.lane_s);
  const factor=flag.neutral?.6:1;
  return {lane:base.lane_s+shift,low:base.lane_low_s+shift,high:base.lane_high_s+shift,
    net:Math.max(0,base.net_loss_s+shift)*factor,netLow:Math.max(0,base.net_low_s+shift)*factor,
    netHigh:Math.max(0,base.net_high_s+shift)*factor,observed:measured.length};
}
function timingContext(){
  const r=state.race,n=state.driver,ownPos=latest(r.positions[n],state.t)?.[1],ownLap=currentLap(n)?.[2];
  const ownInterval=latest(r.intervals[n],state.t),ownGap=ownPos===1?0:numericGap(ownInterval?.[1]);
  if(!ownPos||!ownLap||ownGap==null||(ownPos!==1&&(!ownInterval||state.t-ownInterval[0]>45)))return null;
  const rivals=[];
  for(const driver of r.drivers){
    const number=driver.driver_number;if(number===n||isOut(number)||Math.abs((currentLap(number)?.[2]||0)-ownLap)>1)continue;
    const position=latest(r.positions[number],state.t)?.[1],interval=latest(r.intervals[number],state.t);
    if(!position||!interval||state.t-interval[0]>45)continue;
    const otherGap=position===1?0:numericGap(interval[1]);if(otherGap==null)continue;
    const delta=otherGap-ownGap,stint=currentStint(number,ownLap);
    const tyreAge=stint&&!(stint.current_sample&&stint.tyre_age_at_start==null)?Math.max(0,ownLap-stint.lap_start+(stint.tyre_age_at_start||0)):null;
    const recentPit=r.pits.some(stop=>stop.driver_number===number&&stop.t<=state.t&&state.t-stop.t<180)||r.stints.some(s=>s.driver_number===number&&!s.current_sample&&s.stint_number>1&&s.lap_start>0&&ownLap>=s.lap_start&&ownLap-s.lap_start<=2);
    rivals.push({driver,number,position,delta,relation:delta<0?'AHEAD':'BEHIND',compound:stint?.compound||'UNKNOWN',age:tyreAge,recentPit});
  }
  return {ownPos,ownLap,ownGap,rivals:rivals.sort((a,b)=>Math.abs(a.delta)-Math.abs(b.delta))};
}
function rejoinAnalysis(loss,context){
  if(!context)return null;
  const lostTo=context.rivals.filter(r=>r.delta>0&&r.delta<loss).sort((a,b)=>a.delta-b.delta);
  const firstClear=context.rivals.filter(r=>r.delta>0&&r.delta>=loss).sort((a,b)=>a.delta-b.delta)[0];
  return {position:context.ownPos+lostTo.length,lostTo,margin:firstClear?firstClear.delta-loss:null};
}
function classifyRival(rival,ownAge){
  if(rival.relation==='BEHIND'&&rival.recentPit)return {label:'UNDERCUT',tone:'threat'};
  if(rival.relation==='BEHIND'&&rival.delta<3&&rival.age!=null&&ownAge!=null&&rival.age+3<ownAge)return {label:'UNDERCUT RISK',tone:'threat'};
  if(rival.relation==='AHEAD'&&rival.recentPit)return {label:'OVERCUT CHANCE',tone:'opportunity'};
  if(rival.relation==='AHEAD'&&Math.abs(rival.delta)<5)return {label:'ATTACK',tone:'opportunity'};
  return {label:rival.relation==='AHEAD'?'TARGET':'COVER',tone:''};
}
function renderRivals(context,ownAge){
  const container=$('strategy-rivals');container.replaceChildren();
  const nearby=context?.rivals.filter(r=>Math.abs(r.delta)<=15).slice(0,3)||[];
  txt('rival-summary',nearby.length?`${nearby.length} car${nearby.length===1?'':'s'} inside 15 seconds`:'No comparable nearby timing');
  for(const rival of nearby){
    const status=classifyRival(rival,ownAge),row=document.createElement('div');
    row.className=`rival-row ${status.tone}`;row.style.setProperty('--team',`#${rival.driver.team_colour||'73839b'}`);
    const relation=document.createElement('span'),name=document.createElement('b'),tyre=document.createElement('span'),risk=document.createElement('span');
    relation.className='relation';relation.textContent=rival.relation;name.textContent=`${rival.driver.name_acronym||rival.number} · ${rival.delta<0?'−':'+'}${fixed(Math.abs(rival.delta))}s`;
    tyre.className='rival-tyre';tyre.style.setProperty('--tyre',tyreColors[rival.compound]||tyreColors.UNKNOWN);tyre.textContent=`${rival.compound[0]||'?'} · ${rival.age??'—'}L`;
    risk.className='rival-risk';risk.textContent=status.label;row.append(relation,name,tyre,risk);container.append(row);
  }
  if(!nearby.length){const empty=document.createElement('p');empty.className='empty-message';empty.textContent='Timing gaps will populate after the opening lap.';container.append(empty);}
  return nearby.map(r=>({...r,status:classifyRival(r,ownAge)}));
}
function recordStrategy(action,lap,stopLap,fit,pitLoss,rejoin,sampleAge){
  if(state.payload.mode!=='live'||state.payload.driver!==state.driver)return;
  const key=[state.race.id,state.driver,lap,action,stopLap,fit,rejoin].join('|'),now=performance.now();
  if(state.lastStrategyRecord?.key===key&&now-state.lastStrategyRecord.at<30000)return;
  state.lastStrategyRecord={key,at:now};
  post('streamlit:setComponentValue',{value:{action:'strategy_record',nonce:Date.now()+Math.random(),value:{session_key:state.race.id,driver:state.driver,lap,t:state.t,action,stop_lap:stopLap,fit,pit_loss_s:pitLoss,rejoin_position:rejoin,sample_age_s:sampleAge,input_updated_at:state.race.updated_at,source:state.race.capabilities?.position===false?'free':'openf1'}},dataType:'json'});
}
function updateStrategy(compound,age,lap,history,flag,weather){
  const distanceKnown=state.payload.mode!=='live'||state.race.distance_confirmed;
  const finished=distanceKnown&&(completedLap(state.driver)?.[2]||0)>=state.race.total_laps;
  const dry=['SOFT','MEDIUM','HARD'],remaining=finished?0:Math.max(0,state.race.total_laps-lap+1),forecast=pitForecast(flag),pitLoss=forecast?.net;
  txt('pit-lane',forecast?`${fixed(forecast.lane)} s`:'—');txt('pit-loss',forecast?`${fixed(pitLoss)} s`:'—');
  txt('pit-range',forecast?`Historical 10–90% lane range ${fixed(forecast.low)}–${fixed(forecast.high)} s · nominal loss range ${fixed(forecast.netLow)}–${fixed(forecast.netHigh)} s · ${forecast.observed} earlier routine stops in this race`:'No measured pit-time model available.');
  const context=timingContext(),rivals=renderRivals(context,age),rejoin=forecast?rejoinAnalysis(pitLoss,context):null,low=forecast?rejoinAnalysis(forecast.netLow,context):null,high=forecast?rejoinAnalysis(forecast.netHigh,context):null;
  const threat=rivals.find(r=>r.status.tone==='threat'),opportunity=rivals.find(r=>r.status.label==='OVERCUT CHANCE'),risk=$('pit-risk');
  txt('rejoin-position',rejoin?`P${rejoin.position}`:'—');
  risk.classList.toggle('warning',!!threat||(rejoin&&rejoin.position>context?.ownPos));
  const positionText=rejoin==null?'Rejoin projection waiting for stable timing gaps':`Pit now projects P${rejoin.position}${low&&high?` · range P${low.position}–P${high.position}`:''}${rejoin.margin!=null?` · ${fixed(rejoin.margin)}s clean-air margin`:''}`;
  const battleText=threat?` · ${threat.status.label}: ${threat.driver.name_acronym} ${fixed(Math.abs(threat.delta))}s behind`:opportunity?` · OVERCUT: ${opportunity.driver.name_acronym} has stopped` : '';
  txt('pit-risk',positionText+battleText);
  const timingAge=state.race.health?.timing_age_s,liveAge=state.payload.freshness;
  const ageNow=value=>value==null?null:value+(performance.now()-state.snapshotAt)/1000;
  const selectedAge=ageNow(state.race.capabilities?.position===false?timingAge:state.race.health?.gps_age_s??liveAge),feedAge=ageNow(liveAge);
  const signalAge=state.payload.mode==='live'?(selectedAge==null||feedAge==null?null:Math.max(selectedAge,feedAge)):0;
  const stale=state.payload.mode==='live'&&(state.payload.driver!==state.driver||signalAge==null||signalAge>45);
  const key=[state.race.id,state.driver,lap,compound,age,flag.flag,isOut(state.driver),finished,state.payload.mode,stale,distanceKnown,weather?.rainfall,Math.round((pitLoss||0)*2),rejoin?.position,threat?.number,opportunity?.number].join('|');if(state.planKey===key)return;state.planKey=key;
  $('plan-bar').replaceChildren();
  const unavailable=message=>{txt('strategy-action','Plan unavailable');txt('strategy-detail',message);txt('pit-window','—');txt('next-compound','—');for(const suffix of ['now','plan','late']){txt(`scenario-${suffix}-value`,'—');txt(`scenario-${suffix}-detail`,'Unavailable');}recordStrategy('unavailable',lap,null,null,pitLoss,rejoin?.position,signalAge);};
  const situation=(action,message,fit,window)=>{txt('strategy-action',action);txt('strategy-detail',message);txt('pit-window',window);txt('next-compound',fit);txt('scenario-now-value',action);txt('scenario-now-detail',message);txt('scenario-plan-value',window);txt('scenario-plan-detail',`${fit} transition`);txt('scenario-late-value','REASSESS');txt('scenario-late-detail','Monitor race control and rival stops');recordStrategy(action==='Hold position'?'red_flag':action==='Prepare intermediates'?'weather_inter':'weather_hold',lap,null,fit,pitLoss,rejoin?.position,signalAge);};
  if(stale){unavailable('Fresh race timing is unavailable. Calls are held until the selected feed resumes.');return;}
  if(!distanceKnown){unavailable('Scheduled race distance is not verified for this session. Rival timing remains visible.');return;}
  if(flag.flag==='RED FLAG'){situation('Hold position','Race is suspended. Preserve tyre choice until restart conditions are confirmed.',compound,'RESTART');return;}
  if(pitLoss==null){unavailable('No measured pit-time forecast is available for this race.');return;}
  if(weather?.rainfall&&!['INTERMEDIATE','WET'].includes(compound)){situation('Prepare intermediates','Rain is recorded at the circuit. Confirm grip and field crossover before committing.','INTER',`L${lap}–${Math.min(lap+2,state.race.total_laps)}`);return;}
  if(['INTERMEDIATE','WET'].includes(compound)){situation('Stay weather-aware','Track evolution and rival crossover laps now outweigh the nominal dry plan.',compound,weather?.rainfall?'CURRENT':'DRY CROSSOVER');return;}
  if(age==null){unavailable('Current tyre age is unavailable. Waiting for a usable tyre-state update.');return;}
  if(isOut(state.driver)||remaining===0||remaining>100||!dry.includes(compound)){unavailable(isOut(state.driver)?'The selected car no longer has current lap data.':remaining===0?'Race complete.':'The dry model does not cover this race state.');return;}
  let mask=0;for(const s of history){const idx=dry.indexOf(s.compound);if(idx>=0)mask|=1<<idx;}
  const actions=nominalPlan(remaining,dry.indexOf(compound),age||0,mask,pitLoss),first=actions.findIndex(a=>a>=0);
  const stopLap=first>=0?lap+first:null,next=first>=0?dry[actions[first]]:null,windowStart=stopLap==null?null:Math.max(lap,stopLap-1),windowEnd=stopLap==null?null:Math.min(state.race.total_laps-1,stopLap+2);
  const call=flag.neutral&&first>=0&&first<=2?'Box under neutralisation':threat&&first>=0&&first<=2?`Cover ${threat.driver.name_acronym}`:opportunity&&first>0?'Extend for overcut':first===0?'Pit this lap':first>0?'Stay out':'Run to the flag';
  txt('strategy-action',call);txt('pit-window',stopLap==null?'TO FLAG':`L${windowStart}–${windowEnd}`);
  const confidence=forecast.observed>=3?'race-calibrated':forecast.observed?'blended':'historical-model';
  txt('strategy-detail',first>=0?`Target lap ${stopLap} on ${next.toLowerCase()}. ${confidence} pit loss.${state.payload.weekend?.calibration?' Practice-informed tyre estimate; fuel and traffic remain uncertain.':''}${flag.neutral?' Safety Car/VSC reduces the modeled loss.':''}${threat?' Monitor the car behind.':''}${opportunity?' Watch the car ahead through its out lap.':''}`:'No further stop is optimal in the nominal dry model.');
  txt('next-compound',first>=0?dry[actions[first]]:'—');
  const nowPosition=rejoin?`P${rejoin.position}`:'—';
  txt('scenario-now-value',nowPosition);txt('scenario-now-detail',threat?'Current gaps · undercut threat present':'Current-gap rejoin estimate');
  txt('scenario-plan-value',stopLap==null?'TO FLAG':`LAP ${stopLap}`);txt('scenario-plan-detail',stopLap==null?'No further stop planned':`${next} · future rival gaps unknown`);
  txt('scenario-late-value',stopLap==null?'—':`LAP ${Math.min(stopLap+3,state.race.total_laps-1)}`);txt('scenario-late-detail',stopLap==null?'No extension scenario':'Build tyre offset · reassess rival traffic');
  $('scenario-now').classList.toggle('warning',!!threat);$('scenario-plan').classList.toggle('warning',first<0);
  let c=compound,start=0;for(let i=0;i<=actions.length;i++){if(actions[i]>=0||i===actions.length){const el=document.createElement('span');el.style.width=`${(i-start+1)/Math.max(1,remaining)*100}%`;el.style.setProperty('--tyre',tyreColors[c]);el.title=c;$('plan-bar').append(el);if(i<actions.length)c=dry[actions[i]];start=i+1;}}
  recordStrategy(flag.neutral&&first>=0&&first<=2?'neutralisation':threat&&first>=0&&first<=2?'cover':opportunity&&first>0?'overcut':first===0?'pit_now':first>0?'stay_out':'run_to_flag',lap,stopLap,next,pitLoss,rejoin?.position,signalAge);
}
let bounds=null;
function fitTrack(){const points=state.race.track;if(!points||points.length<2){bounds=null;const canvas=$('track-canvas');canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);return;}bounds={minX:Math.min(...points.map(p=>p[0])),maxX:Math.max(...points.map(p=>p[0])),minY:Math.min(...points.map(p=>p[1])),maxY:Math.max(...points.map(p=>p[1]))};drawTrack();}
function carPoint(number,t){
  const rows=state.race.locations[number]||[];const before=latest(rows,t);if(!before||t-before[0]>10||isOut(number))return null;
  // No interpolation over a missing GPS interval or through a red-flag outage.
  const idx=rows.indexOf(before),after=rows[idx+1];if(!after||after[0]-before[0]>3)return before.slice(1);
  const f=(t-before[0])/(after[0]-before[0]);return [before[1]+(after[1]-before[1])*f,before[2]+(after[2]-before[2])*f];
}
function drawTrack(){
  if(!state.race||!bounds)return;
  const canvas=$('track-canvas'),rect=canvas.getBoundingClientRect();if(!rect.width)return;const dpr=devicePixelRatio||1;
  if(canvas.width!==Math.round(rect.width*dpr)||canvas.height!==Math.round(rect.height*dpr)){canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);}
  const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,rect.width,rect.height);
  const w=rect.width,h=rect.height,pad=55;
  const rotate=w>h*1.35&&(bounds.maxY-bounds.minY)>(bounds.maxX-bounds.minX)*1.25;
  const orient=p=>rotate?[p[1],-p[0]]:p;
  const view=rotate?{minX:bounds.minY,maxX:bounds.maxY,minY:-bounds.maxX,maxY:-bounds.minX}:bounds;
  const scale=Math.min((w-pad*2)/(view.maxX-view.minX),(h-pad*1.65)/(view.maxY-view.minY))*(state.follow?1.65:1);
  const followed=state.follow?carPoint(state.driver,state.t):null,center=followed?orient(followed):[(view.minX+view.maxX)/2,(view.minY+view.maxY)/2];
  const xy=p=>{const q=orient(p);return [w/2+(q[0]-center[0])*scale,h/2-(q[1]-center[1])*scale];};
  function path(points){ctx.beginPath();points.forEach((p,i)=>{const v=xy(p);i?ctx.lineTo(...v):ctx.moveTo(...v);});}
  ctx.lineCap='round';ctx.lineJoin='round';path(state.race.track);ctx.closePath();ctx.strokeStyle='#29323e';ctx.lineWidth=14;ctx.stroke();ctx.strokeStyle='#606b7b';ctx.lineWidth=9;ctx.stroke();ctx.strokeStyle='#1e2630';ctx.lineWidth=6;ctx.stroke();
  const track=state.race.track,first=xy(track[0]),second=xy(track[2]||track[1]),angle=Math.atan2(second[1]-first[1],second[0]-first[0]);
  ctx.save();ctx.translate(...first);ctx.rotate(angle);for(let i=0;i<4;i++)for(let j=0;j<2;j++){ctx.fillStyle=(i+j)%2?'#d2d6de':'#111722';ctx.fillRect(j*3-3,i*3-6,3,3);}ctx.restore();
  for(const sector of state.race.sectors||[]){const p=xy(sector.point);ctx.fillStyle='#1a2230';ctx.beginPath();ctx.arc(p[0],p[1],10,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#3b4a60';ctx.lineWidth=1;ctx.stroke();ctx.fillStyle='#93a5bd';ctx.font='8px Segoe UI';ctx.textAlign='center';ctx.fillText(sector.label,p[0],p[1]+3);}
  // Draw selected car last, keeping its halo visible when the field is bunched.
  const drivers=[...state.race.drivers].sort((a,b)=>(a.driver_number===state.driver)-(b.driver_number===state.driver));
  for(const d of drivers){const point=carPoint(d.driver_number,state.t);if(!point)continue;const p=xy(point),selected=d.driver_number===state.driver,color=`#${d.team_colour||'bac8db'}`;
    if(selected){const trace=(state.race.locations[d.driver_number]||[]).filter(v=>v[0]>state.t-5&&v[0]<=state.t).map(v=>v.slice(1));if(trace.length>1){path(trace);ctx.strokeStyle=color+'80';ctx.lineWidth=3;ctx.stroke();}ctx.beginPath();ctx.arc(p[0],p[1],11,0,Math.PI*2);ctx.strokeStyle=color+'90';ctx.lineWidth=1;ctx.stroke();ctx.fillStyle=color+'15';ctx.fill();}
    ctx.beginPath();ctx.arc(p[0],p[1],selected?5:3.8,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();ctx.strokeStyle=selected?'#fff':'#111620';ctx.lineWidth=selected?1.5:1;ctx.stroke();
    if(state.labels||selected){const text=d.name_acronym||String(d.driver_number),lx=p[0]+9,ly=p[1]-10;ctx.font=`${selected?'700':'600'} 8px Segoe UI`;const tw=ctx.measureText(text).width;ctx.fillStyle=selected?'#252f41ee':'#0e141dd9';ctx.fillRect(lx-3,ly-9,tw+7,14);ctx.fillStyle=selected?'#fff':'#c6cfdd';ctx.textAlign='left';ctx.fillText(text,lx,ly+1);}
  }
}
function frame(now){
  const delta=Math.min((now-state.lastFrame)/1000,.1);state.lastFrame=now;
  if(state.race&&(state.playing||state.payload.mode==='live')){if(state.payload.mode==='live'){state.t=Math.max(0,Math.min(state.race.duration-(state.race.capabilities?.position===false?0:3),state.t+delta));}else if(state.playing){state.t=Math.min(state.race.duration,state.t+delta*state.speed);if(state.t===state.race.duration)state.playing=false;}drawTrack();if(now-state.lastPanel>180){state.lastPanel=now;updatePanels();}if(now-state.lastSave>5000){state.lastSave=now;save();}}
  requestAnimationFrame(frame);
}
$('play').onclick=()=>{if(state.t>=state.race.duration)state.t=0;state.playing=!state.playing;updatePanels();save();};
$('restart').onclick=()=>{state.t=0;state.playing=false;state.planKey=null;updatePanels();save();};
$('seek').oninput=e=>{state.t=e.target.value/1000*state.race.duration;state.planKey=null;updatePanels();drawTrack();save();};
$('playback-speed').onchange=e=>{state.speed=Number(e.target.value);save();};
$('labels').onclick=()=>{state.labels=!state.labels;$('labels').classList.toggle('active',state.labels);$('labels').setAttribute('aria-pressed',String(state.labels));drawTrack();save();};
$('focus').onclick=()=>{state.follow=!state.follow;$('focus').classList.toggle('active',state.follow);$('focus').setAttribute('aria-pressed',String(state.follow));drawTrack();save();};
$('reset-map').onclick=()=>{state.follow=false;$('focus').classList.remove('active');$('focus').setAttribute('aria-pressed','false');drawTrack();save();};
$('session').onchange=e=>send('session',Number(e.target.value));$('season').onchange=e=>send('year',Number(e.target.value));
$('live-mode').onclick=()=>send('mode','live');$('replay-mode').onclick=()=>send('mode','replay');$('return-replay').onclick=()=>send('mode','replay');$('retry').onclick=()=>send('mode',state.payload.mode);$('lab').onclick=()=>send('lab',true);
$('data-note').onclick=()=>$('about-dialog').showModal();$('close-dialog').onclick=()=>$('about-dialog').close();
window.addEventListener('message',e=>{if(e.source===parent&&e.data.type==='streamlit:render')render(e.data.args.payload);});
new ResizeObserver(()=>{height();drawTrack();}).observe(document.body);
post('streamlit:componentReady',{apiVersion:1});height();requestAnimationFrame(frame);
