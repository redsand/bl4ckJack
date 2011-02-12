/****************************************************************************
** Meta object code from reading C++ file 'bl4ckJack_distributed.h'
**
** Created: Sat Feb 12 01:00:07 2011
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../bl4ckJack_distributed.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'bl4ckJack_distributed.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_bl4ckJackBrute[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
       5,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       4,       // signalCount

 // signals: signature, parameters, type, tag, flags
      19,   16,   15,   15, 0x05,
      54,   16,   15,   15, 0x05,
      95,   15,   15,   15, 0x05,
     126,  124,   15,   15, 0x05,

 // slots: signature, parameters, type, tag, flags
     163,   15,   15,   15, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_bl4ckJackBrute[] = {
    "bl4ckJackBrute\0\0,,\0"
    "updateBruteStatus(int,int,QString)\0"
    "updateBruteLabels(double,QString,qint64)\0"
    "updateBruteDateTime(QString)\0,\0"
    "updateBrutePassword(QString,QString)\0"
    "doWork()\0"
};

const QMetaObject bl4ckJackBrute::staticMetaObject = {
    { &QThread::staticMetaObject, qt_meta_stringdata_bl4ckJackBrute,
      qt_meta_data_bl4ckJackBrute, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &bl4ckJackBrute::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *bl4ckJackBrute::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *bl4ckJackBrute::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_bl4ckJackBrute))
        return static_cast<void*>(const_cast< bl4ckJackBrute*>(this));
    return QThread::qt_metacast(_clname);
}

int bl4ckJackBrute::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QThread::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: updateBruteStatus((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< QString(*)>(_a[3]))); break;
        case 1: updateBruteLabels((*reinterpret_cast< double(*)>(_a[1])),(*reinterpret_cast< QString(*)>(_a[2])),(*reinterpret_cast< qint64(*)>(_a[3]))); break;
        case 2: updateBruteDateTime((*reinterpret_cast< QString(*)>(_a[1]))); break;
        case 3: updateBrutePassword((*reinterpret_cast< QString(*)>(_a[1])),(*reinterpret_cast< QString(*)>(_a[2]))); break;
        case 4: doWork(); break;
        default: ;
        }
        _id -= 5;
    }
    return _id;
}

// SIGNAL 0
void bl4ckJackBrute::updateBruteStatus(int _t1, int _t2, QString _t3)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)), const_cast<void*>(reinterpret_cast<const void*>(&_t3)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void bl4ckJackBrute::updateBruteLabels(double _t1, QString _t2, qint64 _t3)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)), const_cast<void*>(reinterpret_cast<const void*>(&_t3)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void bl4ckJackBrute::updateBruteDateTime(QString _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}

// SIGNAL 3
void bl4ckJackBrute::updateBrutePassword(QString _t1, QString _t2)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 3, _a);
}
QT_END_MOC_NAMESPACE
