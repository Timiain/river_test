const canvas = document.getElementById('game');
const ctx = canvas.getContext('2d');

const dropCounter = document.getElementById('drop-count');
const riverCounter = document.getElementById('river-count');

const state = {
  drops: [],
  totalDrops: 0,
  reachedRiver: 0,
};

const terraces = [
  { y: 130, slopeStart: 160, slopeEnd: 640, rise: -40, channelDepth: 16 },
  { y: 220, slopeStart: 130, slopeEnd: 675, rise: -20, channelDepth: 18 },
  { y: 320, slopeStart: 100, slopeEnd: 710, rise: 5, channelDepth: 18 },
  { y: 425, slopeStart: 70, slopeEnd: 750, rise: 20, channelDepth: 20 },
];

function getChannelY(level, x) {
  const t = terraces[level];
  const ratio = Math.min(1, Math.max(0, (x - t.slopeStart) / (t.slopeEnd - t.slopeStart)));
  return t.y + t.rise * ratio;
}

function spawnDrop(x, y) {
  const level = terraces.findIndex((terrace) => y < terrace.y + 25);
  state.drops.push({
    x,
    y,
    level: level === -1 ? terraces.length - 1 : level,
    speed: 1.1 + Math.random() * 0.9,
    drift: 0.5 + Math.random() * 0.8,
    size: 2 + Math.random() * 2,
  });
  state.totalDrops += 1;
}

canvas.addEventListener('click', (event) => {
  const rect = canvas.getBoundingClientRect();
  const x = ((event.clientX - rect.left) / rect.width) * canvas.width;
  const y = ((event.clientY - rect.top) / rect.height) * canvas.height;

  for (let i = 0; i < 8; i += 1) {
    spawnDrop(x + (Math.random() - 0.5) * 40, y + (Math.random() - 0.5) * 24);
  }
});

function update() {
  for (const drop of state.drops) {
    const channelY = getChannelY(drop.level, drop.x);

    if (drop.y < channelY) {
      drop.y += drop.speed;
      drop.x += drop.drift * 0.16;
    } else {
      drop.y = channelY;
      drop.x += drop.drift + drop.level * 0.15;

      if (drop.level < terraces.length - 1) {
        const spillEdge = terraces[drop.level].slopeEnd - 35;
        if (drop.x > spillEdge) {
          drop.level += 1;
          drop.x -= 60;
          drop.y += 16;
        }
      }
    }
  }

  state.drops = state.drops.filter((drop) => {
    if (drop.x > canvas.width - 48 && drop.y > terraces[terraces.length - 1].y) {
      state.reachedRiver += 1;
      return false;
    }
    return drop.x < canvas.width + 20;
  });

  dropCounter.textContent = String(state.totalDrops);
  riverCounter.textContent = String(state.reachedRiver);
}

function drawTerrain() {
  ctx.fillStyle = '#4d7d4a';
  ctx.fillRect(0, 90, canvas.width, canvas.height);

  ctx.fillStyle = '#5f8f58';
  terraces.forEach((terrace, i) => {
    ctx.beginPath();
    ctx.moveTo(0, terrace.y + 60);
    ctx.lineTo(terrace.slopeStart - 20, terrace.y + 40);
    ctx.lineTo(terrace.slopeEnd + 20, terrace.y + terrace.rise + 45);
    ctx.lineTo(canvas.width, terrace.y + terrace.rise + 80 + i * 2);
    ctx.lineTo(canvas.width, canvas.height);
    ctx.lineTo(0, canvas.height);
    ctx.closePath();
    ctx.fill();
  });
}

function drawChannels() {
  ctx.lineCap = 'round';

  terraces.forEach((terrace, i) => {
    const gradient = ctx.createLinearGradient(terrace.slopeStart, terrace.y, terrace.slopeEnd, terrace.y + terrace.rise);
    gradient.addColorStop(0, '#8be6ff');
    gradient.addColorStop(1, '#297bd6');

    ctx.strokeStyle = gradient;
    ctx.lineWidth = terrace.channelDepth;
    ctx.beginPath();
    ctx.moveTo(terrace.slopeStart, terrace.y);
    ctx.lineTo(terrace.slopeEnd, terrace.y + terrace.rise);
    ctx.stroke();

    if (i < terraces.length - 1) {
      ctx.strokeStyle = '#5ec5ff';
      ctx.lineWidth = 8;
      ctx.beginPath();
      ctx.moveTo(terrace.slopeEnd - 20, terrace.y + terrace.rise);
      ctx.quadraticCurveTo(
        terrace.slopeEnd + 10,
        terrace.y + terrace.rise + 25,
        terraces[i + 1].slopeStart + 30,
        terraces[i + 1].y
      );
      ctx.stroke();
    }
  });

  ctx.strokeStyle = '#2f74c7';
  ctx.lineWidth = 30;
  ctx.beginPath();
  ctx.moveTo(canvas.width - 80, terraces[terraces.length - 1].y + 20);
  ctx.lineTo(canvas.width - 5, canvas.height);
  ctx.stroke();
}

function drawDrops() {
  for (const drop of state.drops) {
    ctx.beginPath();
    ctx.fillStyle = '#d9f7ff';
    ctx.arc(drop.x, drop.y, drop.size, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawLabels() {
  ctx.fillStyle = 'rgba(7, 15, 35, 0.58)';
  ctx.fillRect(18, 16, 270, 110);

  ctx.fillStyle = '#e8f7ff';
  ctx.font = '18px sans-serif';
  ctx.fillText('梯级流域示意', 30, 44);
  ctx.font = '14px sans-serif';
  ctx.fillText('· 点击任意位置制造降雨', 30, 72);
  ctx.fillText('· 雨滴沿梯级河道逐级汇流', 30, 96);
  ctx.fillText('· 右侧蓝色主河为出口断面', 30, 120);
}

function render() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawTerrain();
  drawChannels();
  drawDrops();
  drawLabels();
}

function loop() {
  update();
  render();
  requestAnimationFrame(loop);
}

for (let i = 0; i < 45; i += 1) {
  spawnDrop(140 + Math.random() * 140, 80 + Math.random() * 35);
}

loop();
